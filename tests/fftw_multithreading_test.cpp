#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <complex>
#include <cmath>
#include <omp.h>
#include <thread>
#include <algorithm>
#include <iomanip>

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

// FFTW 3D FFT with configurable thread count
double runFFTWMultithread(const std::vector<std::complex<float>>& input,
                       std::vector<std::complex<float>>& output,
                       int dim_x, int dim_y, int dim_z,
                       int num_iterations, int num_threads) {
    
    // Initialize FFTW threading
    if (fftwf_init_threads() == 0) {
        std::cerr << "Failed to initialize FFTW threads!" << std::endl;
        return 0;
    }
    
    // Set the number of threads for FFTW
    fftwf_plan_with_nthreads(num_threads);
    
    // Create FFTW plans with optimized flags
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    
    // Copy input data
    for (size_t i = 0; i < input.size(); ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }
    
    // Use FFTW_MEASURE for optimal performance
    fftwf_plan plan = fftwf_plan_dft_3d(dim_z, dim_y, dim_x,
                                       in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    // Warm-up run
    fftwf_execute(plan);
    
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
    double avg_time = static_cast<double>(duration.count()) / num_iterations;
    
    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    
    std::cout << "FFTW 3D FFT (" << dim_x << "x" << dim_y << "x" << dim_z
              << ") with " << num_threads << " threads:" << std::endl;
    std::cout << "  Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Average time per iteration: " << avg_time << " microseconds" << std::endl;
    std::cout << "  Throughput: " << (1e6 / avg_time) << " FFTs/second" << std::endl;
    return static_cast<double>(duration.count());
}

// Test scaling efficiency
void testScalingEfficiency(const std::vector<std::complex<float>>& input,
                          int dim_x, int dim_y, int dim_z,
                          int num_iterations) {
    
    std::cout << "\n=== Scaling Efficiency Analysis ===" << std::endl;
    
    // Test with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8};
    // thread_counts = std::vector<int>{12};

    
    // Limit to available cores
    int max_threads = std::thread::hardware_concurrency();
    std::cout << "Hardware concurrency: " << max_threads << " threads" << std::endl;
    
    // Filter thread counts to not exceed available cores
    thread_counts.erase(
        std::remove_if(thread_counts.begin(), thread_counts.end(),
                      [max_threads](int count) { return count > max_threads; }),
        thread_counts.end()
    );
    
    std::vector<double> execution_times;
    std::vector<std::complex<float>> result;
    
    for (int threads : thread_counts) {
        std::cout << "\nTesting with " << threads << " thread(s):" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        double duration = runFFTWMultithread(input, result, dim_x, dim_y, dim_z, num_iterations, threads);
        auto end = std::chrono::high_resolution_clock::now();
        
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        execution_times.push_back(duration / num_iterations);
    }
    
    // Calculate and display scaling metrics
    std::cout << "\n=== Scaling Results ===" << std::endl;
    std::cout << "Threads\t\tTime (μs)\tSpeedup\t\tEfficiency" << std::endl;
    std::cout << "-------\t\t---------\t-------\t\t----------" << std::endl;
    
    double baseline_time = execution_times[0]; // Single thread time
    
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        int threads = thread_counts[i];
        double time = execution_times[i];
        double speedup = baseline_time / time;
        double efficiency = speedup / threads * 100.0;
        
        std::cout << threads << "\t\t" << std::fixed << std::setprecision(1) 
                  << time << "\t\t" << std::setprecision(2) << speedup 
                  << "\t\t" << std::setprecision(1) << efficiency << "%" << std::endl;
    }
}

// Memory bandwidth test
void testMemoryBandwidth(int dim_x, int dim_y, int dim_z, int num_threads) {
    std::cout << "\n=== Memory Bandwidth Test ===" << std::endl;
    
    // Generate test data
    std::vector<std::complex<float>> input_data;
    generateRandomData(input_data, dim_x * dim_y * dim_z);
    
    std::vector<std::complex<float>> output;
    
    // Calculate theoretical memory usage
    size_t data_size = dim_x * dim_y * dim_z * sizeof(std::complex<float>);
    size_t total_memory = data_size * 2; // Input + Output
    
    std::cout << "Data size: " << data_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory usage: " << total_memory / (1024 * 1024) << " MB" << std::endl;
    
    // Run test
    auto start = std::chrono::high_resolution_clock::now();
    runFFTWMultithread(input_data, output, dim_x, dim_y, dim_z, 10, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_seconds = duration.count() / 1e6 / 10.0; // Average per iteration
    
    // Calculate bandwidth
    double bandwidth_mbps = (total_memory / (1024 * 1024)) / time_seconds;
    
    std::cout << "Effective memory bandwidth: " << bandwidth_mbps << " MB/s" << std::endl;
}

int main() {
    std::cout << "FFTW Multithreading Performance Test" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Test configurations
    std::vector<std::tuple<int, int, int>> test_sizes = {
        // {64, 64, 64},     // Medium 3D FFT
        {128, 128, 128},  // Large 3D FFT
        {256, 256, 256}   // Very large 3D FFT
    };
    
    int num_iterations = 50;
    
    // Test each size configuration
    for (const auto& size : test_sizes) {
        int dim_x = std::get<0>(size);
        int dim_y = std::get<1>(size);
        int dim_z = std::get<2>(size);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing " << dim_x << "x" << dim_y << "x" << dim_z << " FFT" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Generate random input data
        std::vector<std::complex<float>> input_data;
        generateRandomData(input_data, dim_x * dim_y * dim_z);
        LIKWID_MARKER_INIT;
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_START("Compute");
        // Test scaling efficiency
        testScalingEfficiency(input_data, dim_x, dim_y, dim_z, num_iterations);
        LIKWID_MARKER_STOP("Compute");
        LIKWID_MARKER_CLOSE;
        
        // Test memory bandwidth with optimal thread count
        int optimal_threads = std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
        // testMemoryBandwidth(dim_x, dim_y, dim_z, optimal_threads);
    }
    
    // Cleanup FFTW threading
    fftwf_cleanup_threads();
    
    std::cout << "\nMultithreading test completed!" << std::endl;
    return 0;
}
