#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <future>
#include <functional>
#include <fstream>

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

// FFTW headers
extern "C" {
#include <fftw3.h>
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

// Single large FFTW with multiple threads
double runSingleLargeFFT(const std::vector<std::complex<float>>& input,
                        std::vector<std::complex<float>>& output,
                        int dim_x, int dim_y, int dim_z,
                        int num_iterations, int num_threads) {
    

    
    // Allocate FFTW arrays
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    
    // Copy input data
    for (size_t i = 0; i < input.size(); ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }
    
    // Create plan
    fftwf_plan plan = fftwf_plan_dft_3d(dim_z, dim_y, dim_x,
                                       in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    // Warm-up run
    fftwf_execute(plan);
    
    // Timing runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {

        fftwf_execute_dft(plan, in, out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy output data
    output.resize(dim_x * dim_y * dim_z);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::complex<float>(out[i][0], out[i][1]);
    }
    
    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<double>(duration.count()) / num_iterations;
}



// Multiple smaller FFTs run in parallel
double runMultipleSmallFFTs(const std::vector<std::complex<float>> input,
                           std::vector<std::complex<float>> output,
                           int dim_x, int dim_y, int dim_z, int dim_x_large, int dim_y_large, int dim_z_large,
                           int num_iterations, int num_ffts, int total_threads) {
    
    // Allocate FFTW arrays
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x_large * dim_y_large * dim_z_large);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x_large * dim_y_large * dim_z_large);
    
    // Copy input data
    for (size_t i = 0; i < input.size(); ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }
    
    // Create plan
    fftwf_plan plan = fftwf_plan_dft_3d(dim_z, dim_y, dim_x,
                                       in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    
    // Timing runs
    float duration = 0;
    for (int smallfft = 0; smallfft < num_ffts; smallfft++){

        // fftwf_execute_dft(plan, in + smallfft * 0 * sizeof(fftw_complex), out + smallfft * 0 * sizeof(fftw_complex));
        fftwf_execute(plan);

        int group = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            // fftwf_execute_dft(plan, in + group * i * sizeof(fftw_complex), out + group * i * sizeof(fftw_complex));
            fftwf_execute(plan);
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto durationi = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        duration += static_cast<float>(durationi.count());
    }
    
    
    
    // Copy output data
    output.resize(dim_x * dim_y * dim_z);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::complex<float>(out[i][0], out[i][1]);
    }
    
    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    
    
    return duration / num_iterations;
}

// Test configuration structure
struct TestConfig {
    int large_dim_x, large_dim_y, large_dim_z;  // Single large FFT dimensions
    int small_dim_x, small_dim_y, small_dim_z;  // Individual small FFT dimensions
    int num_small_ffts;                          // Number of small FFTs
    int num_threads;                             // Total number of threads to use
    int num_iterations;                          // Number of iterations for timing
    
    // Calculate total size for verification
    size_t getLargeTotalSize() const {
        return static_cast<size_t>(large_dim_x) * large_dim_y * large_dim_z;
    }
    
    size_t getSmallTotalSize() const {
        return static_cast<size_t>(small_dim_x) * small_dim_y * small_dim_z * num_small_ffts;
    }
    
    bool isValid() const {
        return getLargeTotalSize() == getSmallTotalSize();
    }
};

void calcExtraPaddingWork(const TestConfig& config){

    auto redundancyFactor = [](double L, double s, double p)
    {
        double numerator   = (L / s) * (L / s) * (L / s)
                        * (s + 2.0 * p) * (s + 2.0 * p) * (s + 2.0 * p);

        double denominator = (L + 2.0 * p) * (L + 2.0 * p) * (L + 2.0 * p);

        return numerator / denominator;
    };

    int maxp = 200;
    int stepsize = 20;
    for (int p = 0; p < maxp; p += stepsize){
        double dW = redundancyFactor(static_cast<double>(config.large_dim_x),
                              static_cast<double>(config.small_dim_x),
                              static_cast<double>(p));
        std::cout << "Redundant work for padding " << p << " : " << dW << std::endl;
    }

}
// Comprehensive comparison test
void runDecompositionTest(const TestConfig& config) {
    // if (!config.isValid()) {
    //     std::cerr << "Invalid test configuration: total sizes don't match!" << std::endl;
    //     return;
    // }
    
    std::cout << "\n===========================================" << std::endl;
    std::cout << "FFT Size Decomposition Comparison Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Large FFT: " << config.large_dim_x << "x" << config.large_dim_y 
              << "x" << config.large_dim_z << std::endl;
    std::cout << "Small FFTs: " << config.num_small_ffts << " x (" 
              << config.small_dim_x << "x" << config.small_dim_y 
              << "x" << config.small_dim_z << ")" << std::endl;
    std::cout << "Total size: " << config.getLargeTotalSize() << " complex_t elements" << std::endl;
    std::cout << "Iterations: " << config.num_iterations << std::endl;
    

    // Generate input data for large FFT
    std::vector<std::complex<float>> large_input;
    generateRandomData(large_input, config.getLargeTotalSize());
    
    // Generate input data for small FFTs
    std::vector<std::vector<std::complex<float>>> small_inputs(config.num_small_ffts);
    for (int i = 0; i < config.num_small_ffts; ++i) {
        generateRandomData(small_inputs[i], config.small_dim_x * config.small_dim_y * config.small_dim_z);
    }
    
    std::cout << "\n--- Testing Single Large FFT ---" << std::endl;
    std::vector<std::complex<float>> large_output;
    
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;
    LIKWID_MARKER_START("SingleLargeFFT");
    
    double large_fft_time = runSingleLargeFFT(large_input, large_output,
                                             config.large_dim_x, config.large_dim_y, config.large_dim_z,
                                             config.num_iterations, config.num_threads);
    
    LIKWID_MARKER_STOP("SingleLargeFFT");
    
    std::cout << "Single large FFT time: " << large_fft_time << " μs" << std::endl;
    std::cout << "Throughput: " << (1e6 / large_fft_time) << " FFTs/second" << std::endl;
    
    std::cout << "\n--- Testing Multiple Small FFTs (std::async) ---" << std::endl;
    std::vector<std::vector<std::complex<float>>> small_outputs;
    
    LIKWID_MARKER_START("MultipleSmallFFTs_Async");
    // std::ofstream file;
    // file.open("speed.txt", std::ios_base::app);
    // file << config.large_dim_x << "\t" << large_fft_time << "\n";
    // file.close();
    double small_ffts_time = runMultipleSmallFFTs(large_input, large_output,
                                                config.small_dim_x, config.small_dim_y, config.small_dim_z,
                                                config.large_dim_x, config.large_dim_y, config.large_dim_z,
                                                config.num_iterations, config.num_small_ffts,
                                                config.num_threads);
    
    LIKWID_MARKER_STOP("MultipleSmallFFTs");
    LIKWID_MARKER_CLOSE;
    
    std::cout << "Multiple small FFTs time: " << small_ffts_time << " μs" << std::endl;
    std::cout << "Throughput: " << (1e6 / small_ffts_time) << " batch/second" << std::endl;
    


    // // Calculate and display comparison metrics
    // std::cout << "\n--- Comparison Results ---" << std::endl;
    // double speedup = large_fft_time / small_ffts_time;
    // double efficiency = speedup > 1.0 ? speedup : 1.0 / speedup;
    
    // std::cout << std::fixed << std::setprecision(2);
    // std::cout << "Time ratio (Large/Small): " << speedup << std::endl;
    
    // if (speedup > 1.0) {
    //     std::cout << "Multiple small FFTs are " << speedup << "x faster" << std::endl;
    // } else {
    //     std::cout << "Single large FFT is " << (1.0/speedup) << "x faster" << std::endl;
    // }
    
    // // Memory usage comparison
    // size_t large_memory = config.getLargeTotalSize() * sizeof(std::complex<float>) * 2; // input + output
    // size_t small_memory = config.getSmallTotalSize() * sizeof(std::complex<float>) * 2; // input + output
    
    // std::cout << "\n--- Memory Usage ---" << std::endl;
    // std::cout << "Large FFT memory: " << large_memory / (1024 * 1024) << " MB" << std::endl;
    // std::cout << "Small FFTs memory: " << small_memory / (1024 * 1024) << " MB" << std::endl;
    
    // // Theoretical analysis
    // std::cout << "\n--- Theoretical Analysis ---" << std::endl;
    
    
    // // FFT complexity: O(N log N) where N is the total size
    // double large_complexity = config.getLargeTotalSize() * std::log2(config.getLargeTotalSize());
    // double small_complexity = config.num_small_ffts * 
    //     (config.small_dim_x * config.small_dim_y * config.small_dim_z) *
    //     std::log2(config.small_dim_x * config.small_dim_y * config.small_dim_z);
    
    // std::cout << "Large FFT complexity: " << std::scientific << large_complexity << std::endl;
    // std::cout << "Small FFTs complexity: " << small_complexity << std::endl;
    // std::cout << "Complexity ratio (Small/Large): " << std::fixed << std::setprecision(2) 
    //           << (small_complexity / large_complexity) << std::endl;
    
    // calcExtraPaddingWork(config);
    // fftwf_cleanup_threads();
}

int main() {
    std::cout << "FFTW Size Decomposition Performance Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get system information
    // int max_threads = std::thread::hardware_concurrency();
    // std::cout << "Hardware concurrency: " << max_threads << " threads" << std::endl;
    
    // Test configurations - same total size, different decompositions
    int num_iterations = 20;
    std::vector<TestConfig> test_configs = {
    //     // 128^3 vs 8 x 64^3 (same total size: 2,097,152 elements)

    //     // {512, 512, 512, 256, 256, 256, 8, std::min(8, max_threads), 10},
        {512, 512, 64, 128, 128, 128, 8, 8, num_iterations},
    //     {128, 128, 128, 64, 64, 64, 8, std::min(8, max_threads), num_iterations},
        
    //     // // 256x128x64 vs 4 x 128^3 (same total size: 2,097,152 elements)
    //     // {256, 128, 64, 128, 128, 128, 4, std::min(8, max_threads), 10},
        
    //     // // 512x64x64 vs 16 x 64^3 (same total size: 2,097,152 elements)
    //     // {512, 64, 64, 64, 64, 64, 16, std::min(16, max_threads), 10},
        
    //     // 64^3 vs 8 x 32^3 (smaller test case: 262,144 elements)
    //     {64, 64, 64, 32, 32, 32, 8, std::min(8, max_threads), num_iterations}
    };
    // std::vector<TestConfig> test_configs;
    // for (int i = 260; i < 500; i+=2){
    //     test_configs.emplace_back(TestConfig{i,i,i,i,i,i,1,1, num_iterations});
    // }
    
    // Run all test configurations
    for (const auto& config : test_configs) {
        runDecompositionTest(config);
        std::cout << std::endl;
    }
    
    
    std::cout << "Size decomposition test completed!" << std::endl;
    return 0;
}
