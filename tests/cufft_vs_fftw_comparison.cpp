#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <complex>
#include <cmath>
#include <omp.h>
// FFTW headers
extern "C" {
#include <fftw3.h>
}

// CUDA headers
#include <cuda_runtime.h>
#include <cufft.h>

// Error checking macros
#define CUDA_CHECK(err) { \
    cudaError_t cuda_err = err; \
    if (cuda_err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_err) << " at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CUFFT_CHECK(err) { \
    cufftResult cufft_err = err; \
    if (cufft_err != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error: " << cufft_err << " at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Generate random complex_t data
void generateRandomData(std::vector<std::complex<float>>& data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::complex<float>(dist(gen), dist(gen));
    }
}

// FFTW 3D FFT forward transform (optimized for performance)
void runFFTW3DFFT(const std::vector<std::complex<float>>& input,
                  std::vector<std::complex<float>>& output,
                  int dim_x, int dim_y, int dim_z,
                  int num_iterations) {
    
    // Create FFTW plans with optimized flags
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    
    // Use FFTW_MEASURE for optimal performance (takes longer to plan but faster execution)
    fftwf_plan plan = fftwf_plan_dft_3d(dim_z, dim_y, dim_x,
                                       in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    // Copy input data
    for (size_t i = 0; i < input.size(); ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }
    
    // Run FFTW with warm-up (first execution is slower due to initialization)
    fftwf_execute(plan); // Warm-up run
    
    // Run FFTW for timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        fftwf_execute(plan);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy output data
    output.resize(dim_x * dim_y * dim_z);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::complex<float>(out[i][0], out[i][1]);
    }
    
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    
    std::cout << "FFTW 3D FFT (" << dim_x << "x" << dim_y << "x" << dim_z
              << ") time: " << duration.count() << " microseconds" << std::endl;
}

// cuFFT 3D FFT forward transform (optimized for performance)
void runCuFFT3DFFT(const std::vector<std::complex<float>>& input,
                  std::vector<std::complex<float>>& output,
                  int dim_x, int dim_y, int dim_z,
                  int num_iterations) {
    
    // CUDA device setup
    int device_id;
    CUDA_CHECK(cudaSetDevice(0)); // Use device 0 for optimal performance
    
    // CUDA device properties for optimization
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    
    // Set optimal CUDA configuration
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    
    // Allocate device memory with page-locked host memory for faster transfers
    cufftComplex *d_in, *d_out;
    size_t total_size = dim_x * dim_y * dim_z;
    
    // Use cudaMalloc for device memory (faster than unified memory for this use case)
    CUDA_CHECK(cudaMalloc((void**)&d_in, total_size * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, total_size * sizeof(cufftComplex)));
    
    // Use page-locked host memory for faster host-device transfers
    cufftComplex *h_input;
    CUDA_CHECK(cudaHostAlloc((void**)&h_input, total_size * sizeof(cufftComplex), cudaHostAllocPortable));
    
    // Copy input data to page-locked host memory
    for (size_t i = 0; i < input.size(); ++i) {
        h_input[i].x = input[i].real();
        h_input[i].y = input[i].imag();
    }
    
    // Asynchronous memory copy for better overlap
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_input, total_size * sizeof(cufftComplex),
                              cudaMemcpyHostToDevice, 0));
    
    // Create optimized cuFFT plan
    cufftHandle plan;
    int embed[] = {1, 1, 1}; // No embedding
    int nembed[] = {dim_x, dim_y, dim_z}; // Strided dimensions
    int istride = 1; // Input stride
    int ostride = 1; // Output stride
    
    // Try to create an advanced plan for better performance
    cufftResult result = cufftPlanMany(&plan, 3, &dim_z, // 3D transform
                                      nembed, istride, istride, // Input
                                      nembed, ostride, ostride, // Output
                                      CUFFT_C2C, 1); // Batch size
    
    if (result != CUFFT_SUCCESS) {
        // Fall back to simple plan if advanced plan fails
        CUFFT_CHECK(cufftPlan3d(&plan, dim_z, dim_y, dim_x, CUFFT_C2C));
    }
    
    // Set plan to use CUDA streams for better performance
    CUFFT_CHECK(cufftSetStream(plan, 0));
    
    // Use batch execution for better performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        CUFFT_CHECK(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    }
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy output data back to host (asynchronous)
    cufftComplex *h_output;
    CUDA_CHECK(cudaHostAlloc((void**)&h_output, total_size * sizeof(cufftComplex), cudaHostAllocPortable));
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_out, total_size * sizeof(cufftComplex),
                              cudaMemcpyDeviceToHost, 0));
    
    // Wait for async copy to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy to output vector
    output.resize(total_size);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::complex<float>(h_output[i].x, h_output[i].y);
    }
    
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    
    std::cout << "cuFFT 3D FFT (" << dim_x << "x" << dim_y << "x" << dim_z
              << ") time: " << duration.count() << " microseconds" << std::endl;
}

// Validate FFT results
bool validateResults(const std::vector<std::complex<float>>& fftw_result,
                    const std::vector<std::complex<float>>& cufft_result,
                    float tolerance = 1e-3f) {
    if (fftw_result.size() != cufft_result.size()) {
        return false;
    }
    
    for (size_t i = 0; i < fftw_result.size(); ++i) {
        float diff = std::abs(fftw_result[i] - cufft_result[i]);
        if (diff > tolerance) {
            std::cout << "Validation failed at index " << i << ": FFTW=" << fftw_result[i] 
                      << ", cuFFT=" << cufft_result[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "cuFFT vs FFTW Performance Comparison (Optimized)" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Set OpenMP for FFTW
    omp_set_num_threads(omp_get_max_threads());
    
    // Test configurations
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {32, 32, 32},   // Small 3D FFT
        {64, 64, 64},   // Medium 3D FFT
        {128, 128, 128}, // Large 3D FFT
        {256, 256, 256}  // Very large 3D FFT
    };
    
    int num_iterations = 100;
    
    // Warm-up runs for both libraries to ensure fair comparison
    std::cout << "Performing warm-up runs..." << std::endl;
    std::vector<std::complex<float>> warmup_data(32 * 32 * 32);
    generateRandomData(warmup_data, 32 * 32 * 32);
    
    std::vector<std::complex<float>> warmup_result;
    runFFTW3DFFT(warmup_data, warmup_result, 32, 32, 32, 1);
    runCuFFT3DFFT(warmup_data, warmup_result, 32, 32, 32, 1);
    
    std::cout << "Warm-up completed. Starting main tests..." << std::endl;
    
    for (const auto& size : test_sizes) {
        int dim_x = std::get<0>(size);
        int dim_y = std::get<1>(size);
        int dim_z = std::get<2>(size);
        
        std::cout << "\n--- Testing " << dim_x << "x" << dim_y << "x" << dim_z << " ---" << std::endl;
        
        // Generate random input data
        std::vector<std::complex<float>> input_data;
        generateRandomData(input_data, dim_x * dim_y * dim_z);
        
        // Run FFTW
        std::vector<std::complex<float>> fftw_result;
        runFFTW3DFFT(input_data, fftw_result, dim_x, dim_y, dim_z, num_iterations);
        
        // Run cuFFT
        std::vector<std::complex<float>> cufft_result;
        runCuFFT3DFFT(input_data, cufft_result, dim_x, dim_y, dim_z, num_iterations);
        
        // Validate results
        std::cout << "Validating results... ";
        if (validateResults(fftw_result, cufft_result)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "Comparison completed!" << std::endl;
    return 0;
}