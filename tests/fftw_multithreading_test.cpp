#include <iostream>
#include <thread>
#include <functional>
#include <future>
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

// Configuration struct for FFTW tests
struct FFTWConfig {
    int dim_x = 0;
    int dim_y = 0;
    int dim_z = 0;
    int num_iterations = 1;
    int num_threads = 1; // default threads to use for functions that accept a single value
    std::vector<int> thread_counts; // optional list of thread counts to test scaling
};

// FFTW 3D FFT with configurable thread count
double runFFTWMultithreadSeperate(const std::vector<std::complex<float>>& input,
                       std::vector<std::complex<float>>& output,
                       const FFTWConfig& cfg) {
    int dim_x = cfg.dim_x;
    int dim_y = cfg.dim_y;
    int dim_z = cfg.dim_z;
    if(cfg.num_threads == 8){
        dim_x /=2;
        dim_y /=2;
        dim_z /=2;
    }
    // Initialize FFTW threading
    if (fftwf_init_threads() == 0) {
        std::cerr << "Failed to initialize FFTW threads!" << std::endl;
        return 0;
    }
    // Set the number of threads for FFTW
    // In this "separate" variant we execute independent plans in parallel using std::async,
    // so keep FFTW single-threaded per plan to avoid oversubscription.
    
        // Use FFTW_MEASURE for optimal performance
    // Create FFTW plans with optimized flags
    fftwf_complex* in_g = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
    fftwf_complex* out_g = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);

        
    fftwf_plan_with_nthreads(1);
    fftwf_plan plan = fftwf_plan_dft_3d(dim_z, dim_y, dim_x,
                                    in_g, out_g, FFTW_FORWARD, FFTW_MEASURE);



    std::function<double(int)> singleFFT([&](int thread_id){
        // Warm-up run



        // Allocate FFTW arrays
        fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
        fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * dim_x * dim_y * dim_z);
        
        // Copy input data
        for (size_t i = 0; i < dim_x * dim_y * dim_z; ++i) {
            in[i][0] = input[i].real();
            in[i][1] = input[i].imag();
        }
        

        // fftwf_execute_dft(plan, in, out);

        
        // Run FFTW for timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cfg.num_iterations; ++i) {
            fftwf_execute_dft(plan, in, out);
        }
        auto end = std::chrono::high_resolution_clock::now();
        

        
        // Calculate duration
        auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        fftwf_free(in);
        fftwf_free(out);
        
        
        return duration;
    });

    std::vector<std::future<double>> runs;
    double duration = 0;

    int runners = cfg.num_threads; // number of parallel tasks to spawn
    for (int i = 0; i < runners; ++i) {
        runs.emplace_back(std::async(std::launch::async, singleFFT, i));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (auto& job : runs) {
        duration += job.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());


    double avg_time = duration / cfg.num_iterations;

    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
    fftwf_cleanup();

    std::cout << "FFTW 3D FFT seperate (" << dim_x << "x" << dim_y << "x" << dim_z
              << ") with " << cfg.num_threads << " threads (" << runners << " parallel plans):" << std::endl;
    std::cout << "  Total time: " << duration << " microseconds" << std::endl;
    std::cout << "  Average time per iteration: " << avg_time << " microseconds" << std::endl;
    std::cout << "  Throughput: " << (1e6 / avg_time) << " FFTs/second" << std::endl;
    return static_cast<double>(duration);
}

// FFTW 3D FFT with configurable thread count
double runFFTWMultithread(const std::vector<std::complex<float>>& input,
                       std::vector<std::complex<float>>& output,
                       const FFTWConfig& cfg) {
     
     // Initialize FFTW threading
     if (fftwf_init_threads() == 0) {
         std::cerr << "Failed to initialize FFTW threads!" << std::endl;
         return 0;
     }
     
     // Set the number of threads for FFTW
    fftwf_plan_with_nthreads(cfg.num_threads);
         // Use FFTW_MEASURE for optimal performance
    // Create FFTW plans with optimized flags
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * cfg.dim_x * cfg.dim_y * cfg.dim_z);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * cfg.dim_x * cfg.dim_y * cfg.dim_z);
    
    fftwf_plan plan = fftwf_plan_dft_3d(cfg.dim_z, cfg.dim_y, cfg.dim_x,
                                       in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    auto start = std::chrono::high_resolution_clock::now();

    // Copy input data
    for (size_t i = 0; i < input.size(); ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }
    

    // Warm-up run
    // fftwf_execute_dft(plan, in, out);
    
    // Run FFTW for timing
    
    for (int i = 0; i < cfg.num_iterations; ++i) {
        fftwf_execute_dft(plan, in, out);
    }
    
    

    

    // Cleanup
    
    fftwf_free(in);
    fftwf_free(out);

    auto end = std::chrono::high_resolution_clock::now();
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time = static_cast<double>(duration.count()) / cfg.num_iterations;
    
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
    fftwf_cleanup();
    
    std::cout << "FFTW 3D FFT (" << cfg.dim_x << "x" << cfg.dim_y << "x" << cfg.dim_z
              << ") with " << cfg.num_threads << " threads:" << std::endl;
    std::cout << "  Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Average time per iteration: " << avg_time << " microseconds" << std::endl;
    std::cout << "  Throughput: " << (1e6 / avg_time) << " FFTs/second" << std::endl;
    return static_cast<double>(duration.count());
}

// Test scaling efficiency
void testScalingEfficiency(const std::vector<std::complex<float>>& input,
                          const FFTWConfig& cfg) {
    
    std::cout << "\n=== Scaling Efficiency Analysis ===" << std::endl;
    
    // Test with different thread counts
    std::vector<int> thread_counts = {1, 8};
    if (!cfg.thread_counts.empty()) {
        thread_counts = cfg.thread_counts;
    }
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
    std::vector<double> execution_times_seperate;

    std::vector<std::complex<float>> result;
    
    for (int threads : thread_counts) {
        std::cout << "\nTesting with " << threads << " thread(s):" << std::endl;
        
        FFTWConfig local_cfg = cfg;
        local_cfg.num_threads = threads;
        
        double duration = runFFTWMultithread(input, result, local_cfg);
         
        double duration_seperate = runFFTWMultithreadSeperate(input, result, local_cfg);
        
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        execution_times.push_back(duration / cfg.num_iterations);
        execution_times_seperate.push_back(duration_seperate / cfg.num_iterations);
    }
    
    // Calculate and display scaling metrics
    std::cout << "\n=== Scaling Results ===" << std::endl;
    std::cout << "Threads\t\tfftwMultiThread (μs)\t\tfftwSeperateCubes\t\tEfficiency" << std::endl;
    std::cout << "-------\t\t---------\t-------\t\t----------" << std::endl;
    
    double baseline_time = execution_times[0]; // Single thread time
    
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        int threads = thread_counts[i];
        double time = execution_times[i];
        double time_seperate = execution_times_seperate[i];
        
        std::cout << threads << "\t\t" << std::fixed << std::setprecision(1) 
                  << time << "\t\t" << time_seperate << "\t\t" << std::setprecision(2) << std::endl;
    }
}

// Memory bandwidth test
void testMemoryBandwidth(const FFTWConfig& cfg) {
    std::cout << "\n=== Memory Bandwidth Test ===" << std::endl;
    
    // Generate test data
    std::vector<std::complex<float>> input_data;
    generateRandomData(input_data, cfg.dim_x * cfg.dim_y * cfg.dim_z);
    
    std::vector<std::complex<float>> output;
    
    // Calculate theoretical memory usage
    size_t data_size = cfg.dim_x * cfg.dim_y * cfg.dim_z * sizeof(std::complex<float>);
    size_t total_memory = data_size * 2; // Input + Output
    
    std::cout << "Data size: " << data_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory usage: " << total_memory / (1024 * 1024) << " MB" << std::endl;
    
    // Run test
    auto start = std::chrono::high_resolution_clock::now();
    FFTWConfig local = cfg;
    local.num_iterations = 10;
    runFFTWMultithread(input_data, output, local);
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
    
    // Test configurations: create a vector of FFTWConfig entries
    int num_iterations = 50;
    std::vector<FFTWConfig> configs;

    // populate configs for the sizes we want to test
    {
        FFTWConfig c;
        c.dim_x = 512; c.dim_y = 512; c.dim_z = 64; c.num_iterations = num_iterations;
        configs.push_back(c);
    }
    {
        FFTWConfig c;
        c.dim_x = 128; c.dim_y = 128; c.dim_z = 128; c.num_iterations = num_iterations;
        configs.push_back(c);
    }
    {
        FFTWConfig c;
        c.dim_x = 256; c.dim_y = 256; c.dim_z = 256; c.num_iterations = num_iterations;
        configs.push_back(c);
    }

    // Test each configuration
    for (const auto& cfg : configs) {
        int dim_x = cfg.dim_x;
        int dim_y = cfg.dim_y;
        int dim_z = cfg.dim_z;

        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing " << dim_x << "x" << dim_y << "x" << dim_z << " FFT" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Generate random input data
        std::vector<std::complex<float>> input_data;
        generateRandomData(input_data, dim_x * dim_y * dim_z);
        LIKWID_MARKER_INIT;
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_START("Compute");
        // Test scaling efficiency using the configuration from the configs vector
        testScalingEfficiency(input_data, cfg);
        LIKWID_MARKER_STOP("Compute");
        LIKWID_MARKER_CLOSE;
        
        // Test memory bandwidth with optimal thread count
        int optimal_threads = std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
        // testMemoryBandwidth using config
        // cfg.num_threads = optimal_threads;
        // testMemoryBandwidth(cfg);
    }
    
    // Cleanup FFTW threading
    
    std::cout << "\nMultithreading test completed!" << std::endl;
    return 0;
}
