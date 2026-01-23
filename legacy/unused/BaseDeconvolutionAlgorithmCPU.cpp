#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h"
#include "UtlFFT.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <thread>
#include <fstream>
#include <sstream>
#include <cmath>

#include <omp.h>

// Memory allocation guard macros
#define ALLOCATE_MEMORY(ptr, size) \
    do { \
        ptr = (complex_t*)fftw_malloc(sizeof(complex_t) * (size)); \
        if (ptr == nullptr) { \
            std::cerr << "[ERROR] Failed to allocate memory for " << #ptr << std::endl; \
            return false; \
        } \
    } while(0)

#define DEALLOCATE_MEMORY(ptr) 
    do { \
        if (ptr != nullptr) { \
            fftw_free(ptr); \
            ptr = nullptr; \
        } \
    } while(0)

// Error handling macros
#define FFTW_CHECK(plan, operation) \
    do { \
        if (!validateFFTWPlan(plan)) { \
            logFFTWError(plan, operation); \
            return false; \
        } \
    } while(0)

BaseDeconvolutionAlgorithmCPU::BaseDeconvolutionAlgorithmCPU()
    : DeconvolutionProcessor()
    , m_forwardPlan(nullptr)
    , m_backwardPlan(nullptr)
    , m_fftwInitialized(false)
    , m_optimizePlans(true)
    , m_fftwThreads(1)
{
    std::cout << "[INFO] BaseDeconvolutionAlgorithmCPU constructor" << std::endl;
    
    // Initialize FFTW thread environment
    setupThreadedFFTW();
}

BaseDeconvolutionAlgorithmCPU::~BaseDeconvolutionAlgorithmCPU() {
    cleanup();
    cleanupBackendSpecific();
}

void BaseDeconvolutionAlgorithmCPU::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    std::cout << "[CONFIGURATION] BaseDeconvolutionAlgorithmCPU configuration" << std::endl;
    
    // Configure FFTW optimization settings
    m_optimizePlans = config.time;  // Use timing flag as optimization indicator
    m_fftwThreads = omp_get_max_threads();
    
    std::cout << "[CONFIGURATION] FFTW optimization: " << (m_optimizePlans ? "enabled" : "disabled") << std::endl;
    std::cout << "[CONFIGURATION] FFTW threads: " << m_fftwThreads << std::endl;
    
    // Set up FFTW plans
    if (!createFFTWPlans()) {
        std::cerr << "[ERROR] Failed to create FFTW plans during configuration" << std::endl;
        return;
    }
}

bool BaseDeconvolutionAlgorithmCPU::preprocessBackendSpecific(int channel_num, int psf_index) {
    std::cout << "[STATUS] CPU preprocessing for channel " << channel_num 
              << " with PSF " << psf_index << std::endl;
    
    try {
        // Allocate channel-specific memory if needed
        if (!manageChannelSpecificMemory(channel_num)) {
            std::cerr << "[ERROR] Failed to manage memory for channel " << channel_num << std::endl;
            return false;
        }
        
        // Validate memory allocation
        if (m_channelSpecificMemory.empty() || 
            m_channelSpecificMemory[channel_num].empty()) {
            std::cerr << "[ERROR] Channel-specific memory not properly allocated" << std::endl;
            return false;
        }
        
        // Log memory usage
        logMemoryUsage("preprocessBackendSpecific");
        
        std::cout << "[STATUS] CPU preprocessing completed for channel " << channel_num << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CPU preprocessing: " << e.what() << std::endl;
        return false;
    }
}

void BaseDeconvolutionAlgorithmCPU::algorithmBackendSpecific(int channel_num, complex_t* H, complex_t* g, complex_t* f) {
    std::cout << "[STATUS] CPU algorithm processing for channel " << channel_num << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        if (!validateComplexArray(H, cubeVolume, "PSF")) {
            std::cerr << "[ERROR] Invalid PSF array in CPU algorithm" << std::endl;
            return;
        }
        
        if (!validateComplexArray(g, cubeVolume, "Input image")) {
            std::cerr << "[ERROR] Invalid input image array in CPU algorithm" << std::endl;
            return;
        }
        
        // Create temporary arrays for processing
        complex_t* temp_g = nullptr;
        complex_t* temp_f = nullptr;
        
        if (!allocateCPUArray(temp_g, cubeVolume) || 
            !allocateCPUArray(temp_f, cubeVolume)) {
            std::cerr << "[ERROR] Failed to allocate temporary arrays for CPU processing" << std::endl;
            DEALLOCATE_MEMORY(temp_g);
            DEALLOCATE_MEMORY(temp_f);
            return;
        }
        
        // Copy input data to working arrays
        std::memcpy(temp_g, g, sizeof(complex_t) * cubeVolume);
        std::memcpy(temp_f, g, sizeof(complex_t) * cubeVolume);  // Initialize with input data
        
        // Execute main algorithm using FFTW utilities
        // Note: iterations would be defined in the concrete algorithm class
        // For now, using a hardcoded iteration count for the base implementation
        const int MAX_ITERATIONS = 10; // Temporary default iteration count
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            std::cout << "\r[STATUS] Channel: " << channel_num + 1 << " Iteration: " << iter + 1
                      << "/" << MAX_ITERATIONS << " ";
            
            // Step 1: Forward FFT of current estimate
            if (!executeForwardFFT(temp_f, temp_g)) {
                std::cerr << "[ERROR] Forward FFT failed in CPU algorithm" << std::endl;
                break;
            }
            
            // Step 2: Multiply with PSF in frequency domain
            UtlFFT::complexMultiplication(temp_f, H, temp_g, cubeVolume);
            
            // Step 3: Backward FFT
            if (!executeBackwardFFT(temp_g, temp_f)) {
                std::cerr << "[ERROR] Backward FFT failed in CPU algorithm" << std::endl;
                break;
            }
            
            // Step 4: Normalize and shift
            UtlFFT::octantFourierShift(temp_f, cubeWidth, cubeHeight, cubeDepth);
            
            // Step 5: Calculate correction factor
            UtlFFT::complexDivision(g, temp_f, temp_g, cubeVolume, epsilon);
            
            // Step 6: Forward FFT of correction factor
            if (!executeForwardFFT(temp_g, temp_f)) {
                std::cerr << "[ERROR] Forward FFT of correction factor failed" << std::endl;
                break;
            }
            
            // Step 7: Multiply with conjugate PSF
            UtlFFT::complexMultiplicationWithConjugate(temp_f, H, temp_f, cubeVolume);
            
            // Step 8: Backward FFT
            if (!executeBackwardFFT(temp_f, temp_g)) {
                std::cerr << "[ERROR] Backward FFT of correction failed" << std::endl;
                break;
            }
            
            // Step 9: Shift and normalize
            UtlFFT::octantFourierShift(temp_g, cubeWidth, cubeHeight, cubeDepth);
            
            // Step 10: Update estimate
            std::memcpy(temp_f, temp_g, sizeof(complex_t) * cubeVolume);
            
            // Validate result
            if (!validateComplexArray(temp_f, cubeVolume, "Iteration result")) {
                std::cerr << "[WARNING] Invalid array values detected in iteration " << iter + 1 << std::endl;
            }
        }
        
        // Copy final result
        std::memcpy(f, temp_f, sizeof(fftw_complex) * cubeVolume);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        m_executionTimes.push_back(duration.count());
        
        std::cout << std::endl;
        std::cout << "[INFO] CPU algorithm processing completed in " << duration.count() << " ms" << std::endl;
        
        // Cleanup temporary arrays
        DEALLOCATE_MEMORY(temp_g);
        DEALLOCATE_MEMORY(temp_f);
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CPU algorithm: " << e.what() << std::endl;
    }
}

bool BaseDeconvolutionAlgorithmCPU::postprocessBackendSpecific(int channel_num, int psf_index) {
    std::cout << "[STATUS] CPU postprocessing for channel " << channel_num << std::endl;
    
    try {
        // Clean up channel-specific memory
        cleanupChannelMemory(channel_num);
        
        // Generate performance metrics
        logPerformanceMetrics();
        
        logMemoryUsage("postprocessBackendSpecific");
        
        std::cout << "[STATUS] CPU postprocessing completed for channel " << channel_num << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CPU postprocessing: " << e.what() << std::endl;
        return false;
    }
}

bool BaseDeconvolutionAlgorithmCPU::allocateBackendMemory(int channel_num) {
    std::cout << "[STATUS] Allocating backend memory for channel " << channel_num << std::endl;
    
    bool success = manageChannelSpecificMemory(channel_num);
    
    if (success) {
        std::cout << "[STATUS] Backend memory allocation successful for channel " << channel_num << std::endl;
    } else {
        std::cerr << "[ERROR] Backend memory allocation failed for channel " << channel_num << std::endl;
    }
    
    return success;
}

void BaseDeconvolutionAlgorithmCPU::deallocateBackendMemory(int channel_num) {
    std::cout << "[STATUS] Deallocating backend memory for channel " << channel_num << std::endl;
    
    cleanupChannelMemory(channel_num);
    
    std::cout << "[STATUS] Backend memory deallocation completed for channel " << channel_num << std::endl;
}

void BaseDeconvolutionAlgorithmCPU::cleanupBackendSpecific() {
    std::cout << "[STATUS] Cleaning up CPU-specific resources" << std::endl;
    
    destroyFFTWPlans();
    
    // Clean up all allocated arrays
    for (fftw_complex* array : m_allocatedArrays) {
        DEALLOCATE_MEMORY(array);
    }
    m_allocatedArrays.clear();
    
    // Clean up channel-specific memory
    for (auto& channel_arrays : m_channelSpecificMemory) {
        for (fftw_complex* array : channel_arrays) {
            DEALLOCATE_MEMORY(array);
        }
        channel_arrays.clear();
    }
    m_channelSpecificMemory.clear();
    
    m_fftwInitialized = false;
    
    std::cout << "[STATUS] CPU-specific cleanup completed" << std::endl;
}

// Protected helper methods

bool BaseDeconvolutionAlgorithmCPU::createFFTWPlans() {
    std::cout << "[STATUS] Creating FFTW plans..." << std::endl;
    
    if (m_fftwInitialized) {
        destroyFFTWPlans();
    }
    
    try {
        // Create forward plan
        int flags = m_optimizePlans ? FFTW_MEASURE : FFTW_ESTIMATE;
        m_forwardPlan = fftw_plan_dft_3d(cubeDepth, cubeHeight, cubeWidth,
                                       nullptr, nullptr, // Input/output arrays managed externally
                                       FFTW_FORWARD, flags);
        
        FFTW_CHECK(m_forwardPlan, "Forward FFT plan creation");
        
        // Create backward plan
        m_backwardPlan = fftw_plan_dft_3d(cubeDepth, cubeHeight, cubeWidth,
                                        nullptr, nullptr, // Input/output arrays managed externally
                                        FFTW_BACKWARD, flags);
        
        FFTW_CHECK(m_backwardPlan, "Backward FFT plan creation");
        
        m_fftwInitialized = true;
        optimizeFFTWPlans();
        
        std::cout << "[STATUS] FFTW plans created successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to create FFTW plans: " << e.what() << std::endl;
        return false;
    }
}

void BaseDeconvolutionAlgorithmCPU::destroyFFTWPlans() {
    if (m_forwardPlan != nullptr) {
        fftw_destroy_plan(m_forwardPlan);
        m_forwardPlan = nullptr;
    }
    
    if (m_backwardPlan != nullptr) {
        fftw_destroy_plan(m_backwardPlan);
        m_backwardPlan = nullptr;
    }
    
    m_fftwInitialized = false;
}

bool BaseDeconvolutionAlgorithmCPU::executeForwardFFT(fftw_complex* input, fftw_complex* output) {
    if (!validateFFTWPlan(m_forwardPlan)) {
        return false;
    }
    
    std::memcpy(input, output, sizeof(fftw_complex) * cubeVolume);
    fftw_execute_dft(m_forwardPlan, input, output);
    
    return true;
}

bool BaseDeconvolutionAlgorithmCPU::executeBackwardFFT(fftw_complex* input, fftw_complex* output) {
    if (!validateFFTWPlan(m_backwardPlan)) {
        return false;
    }
    
    std::memcpy(input, output, sizeof(fftw_complex) * cubeVolume);
    fftw_execute_dft(m_backwardPlan, input, output);
    
    return true;
}

bool BaseDeconvolutionAlgorithmCPU::validateFFTWPlan(fftw_plan plan) {
    return plan != nullptr;
}

bool BaseDeconvolutionAlgorithmCPU::allocateCPUArray(fftw_complex*& array, size_t size) {
    if (size == 0) {
        std::cerr << "[ERROR] Requested zero-size memory allocation" << std::endl;
        return false;
    }
    
    if (!validateMemoryAllocation(size)) {
        std::cerr << "[ERROR] Memory validation failed for size " << size << std::endl;
        return false;
    }
    
    ALLOCATE_MEMORY(array, size);
    
    if (array != nullptr) {
        m_allocatedArrays.push_back(array);
        return true;
    }
    
    return false;
}

void BaseDeconvolutionAlgorithmCPU::deallocateCPUArray(fftw_complex* array) {
    auto it = std::find(m_allocatedArrays.begin(), m_allocatedArrays.end(), array);
    if (it != m_allocatedArrays.end()) {
        DEALLOCATE_MEMORY(array);
        m_allocatedArrays.erase(it);
    }
}

bool BaseDeconvolutionAlgorithmCPU::manageChannelSpecificMemory(int channel_num) {
    // Ensure we have enough space for this channel
    while (m_channelSpecificMemory.size() <= static_cast<size_t>(channel_num)) {
        m_channelSpecificMemory.emplace_back();
    }
    
    // Clean up existing memory for this channel
    for (fftw_complex* array : m_channelSpecificMemory[channel_num]) {
        DEALLOCATE_MEMORY(array);
    }
    m_channelSpecificMemory[channel_num].clear();
    
    // Allocate new memory arrays for this channel
    fftw_complex* channel_array = nullptr;
    if (!allocateCPUArray(channel_array, cubeVolume)) {
        return false;
    }
    
    m_channelSpecificMemory[channel_num].push_back(channel_array);
    
    return true;
}

bool BaseDeconvolutionAlgorithmCPU::validateComplexArray(fftw_complex* array, size_t size, const std::string& array_name) {
    if (array == nullptr) {
        std::cerr << "[ERROR] Null array detected: " << array_name << std::endl;
        return false;
    }
    
    if (size == 0) {
        std::cerr << "[ERROR] Zero-sized array detected: " << array_name << std::endl;
        return false;
    }
    
    bool has_invalid_values = false;
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(array[i][0]) || !std::isfinite(array[i][1])) {
            has_invalid_values = true;
            array[i][0] = 0.0;  // Reset invalid values
            array[i][1] = 0.0;
        }
    }
    
    if (has_invalid_values) {
        std::cerr << "[WARNING] Invalid (NaN/Inf) values detected and reset in: " << array_name << std::endl;
    }
    
    return !has_invalid_values;
}

bool BaseDeconvolutionAlgorithmCPU::normalizeComplexArray(fftw_complex* array, size_t size, double epsilon) {
    if (!validateComplexArray(array, size, "normalizeComplexArray")) {
        return false;
    }
    
    double max_real = 0.0;
    double max_imag = 0.0;
    
    // Find maximum absolute values
    for (size_t i = 0; i < size; ++i) {
        max_real = std::max(max_real, std::abs(array[i][0]));
        max_imag = std::max(max_imag, std::abs(array[i][1]));
    }
    
    // Normalize with epsilon protection
    double norm_factor_real = max_real > epsilon ? (1.0 / max_real) : 1.0;
    double norm_factor_imag = max_imag > epsilon ? (1.0 / max_imag) : 1.0;
    
    for (size_t i = 0; i < size; ++i) {
        array[i][0] *= norm_factor_real;
        array[i][1] *= norm_factor_imag;
    }
    
    return true;
}

bool BaseDeconvolutionAlgorithmCPU::copyComplexArray(const fftw_complex* source, fftw_complex* destination, size_t size) {
    if (!validateComplexArray(const_cast<fftw_complex*>(source), size, "copyComplexArray source")) {
        return false;
    }
    
    std::memcpy(destination, source, sizeof(fftw_complex) * size);
    return true;
}

void BaseDeconvolutionAlgorithmCPU::logFFTWError(fftw_plan plan, const std::string& operation) {
    std::cerr << "[ERROR] FFTW plan failed for operation: " << operation << std::endl;
    std::cerr << "[ERROR] Plan address: " << plan << std::endl;
}

bool BaseDeconvolutionAlgorithmCPU::checkMemoryCriticalSystem() const {
    // Simple check for memory-critical conditions
    long total_memory = 0;
    //估算使用的内存
    for (const auto& channel_arrays : m_channelSpecificMemory) {
        for (const fftw_complex* array : channel_arrays) {
            total_memory += cubeVolume * sizeof(fftw_complex);
        }
    }
    
    // Simple threshold check
    const size_t memory_threshold = 1024ULL * 1024 * 1024 * 2; // 2GB threshold
    return total_memory > memory_threshold;
}

void BaseDeconvolutionAlgorithmCPU::logMemoryUsage(const std::string& operation) const {
    size_t total_allocated = 0;
    for (const auto& channel_arrays : m_channelSpecificMemory) {
        for (const fftw_complex* array : channel_arrays) {
            total_allocated += cubeVolume * sizeof(fftw_complex);
        }
    }
    
    std::cout << "[MEMORY] Operation '" << operation << "': ";
    std::cout << total_allocated / (1024.0 * 1024.0) << " MB allocated" << std::endl;
}

void BaseDeconvolutionAlgorithmCPU::optimizeFFTWPlans() {
    if (m_fftwInitialized && m_optimizePlans) {
        std::cout << "[STATUS] Optimizing FFTW plans with " << m_fftwThreads << " threads..." << std::endl;
        
        // Use multithreaded FFTW if available
        if (fftw_init_threads() == 1) {
            fftw_plan_with_nthreads(m_fftwThreads);
            std::cout << "[STATUS] FFTW threads optimized for " << m_fftwThreads << " cores" << std::endl;
        } else {
            std::cerr << "[WARNING] Failed to initialize FFTW threads" << std::endl;
        }
    }
}

void BaseDeconvolutionAlgorithmCPU::setupThreadedFFTW() {
    // Initialize FFTW threading
    if (fftw_init_threads() == 1) {
        fftw_make_planner_thread_safe();
        std::cout << "[STATUS] FFTW thread environment initialized" << std::endl;
    } else {
        std::cerr << "[WARNING] Failed to initialize FFTW threading" << std::endl;
    }
}

bool BaseDeconvolutionAlgorithmCPU::setupFFTWThreadEnvironment() {
    if (!m_fftwInitialized) {
        if (fftw_init_threads() == 1) {
            fftw_plan_with_nthreads(m_fftwThreads);
            fftw_make_planner_thread_safe();
            m_fftwInitialized = true;
            return true;
        }
        return false;
    }
    return true;
}

bool BaseDeconvolutionAlgorithmCPU::validateMemoryAllocation(size_t required_size) {
    // Simple validation to prevent extreme allocations
    const size_t max_allocation = 10ULL * 1024 * 1024 * 1024; // 10GB limit
    if (required_size > max_allocation) {
        std::cerr << "[ERROR] Requested memory size (" << required_size 
                  << ") exceeds maximum allowed (" << max_allocation << ")" << std::endl;
        return false;
    }
    
    // Check system memory availability
    if (checkMemoryCriticalSystem()) {
        std::cerr << "[WARNING] System memory is critically low - allocations may fail" << std::endl;
    }
    
    return true;
}

void BaseDeconvolutionAlgorithmCPU::cleanupChannelMemory(int channel_num) {
    if (channel_num >= 0 && channel_num < m_channelSpecificMemory.size()) {
        for (fftw_complex* array : m_channelSpecificMemory[channel_num]) {
            DEALLOCATE_MEMORY(array);
        }
        m_channelSpecificMemory[channel_num].clear();
    }
}

void BaseDeconvolutionAlgorithmCPU::logPerformanceMetrics() {
    if (m_executionTimes.empty()) {
        return;
    }
    
    double avg_time = 0.0;
    double min_time = m_executionTimes[0];
    double max_time = m_executionTimes[0];
    
    for (double time : m_executionTimes) {
        avg_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    avg_time /= m_executionTimes.size();
    
    std::cout << "[PERFORMANCE] Algorithm execution times:" << std::endl;
    std::cout << "[PERFORMANCE]  Average: " << avg_time << " ms" << std::endl;
    std::cout << "[PERFORMANCE]  Minimum: " << min_time << " ms" << std::endl;
    std::cout << "[PERFORMANCE]  Maximum: " << max_time << " ms" << std::endl;
    std::cout << "[PERFORMANCE]  Total operations: " << m_executionTimes.size() << std::endl;
}

bool BaseDeconvolutionAlgorithmCPU::handleFFTWError(fftw_plan plan, const std::string& operation) {
    if (!validateFFTWPlan(plan)) {
        logFFTWError(plan, operation);
        return false;
    }
    return true;
}

std::string BaseDeconvolutionAlgorithmCPU::getFFTWErrorString(int error_code) const {
    // Convert FFTW error codes to readable strings
    switch (error_code) {
        case 1: return "Invalid parameters";
        case 2: return "Memory allocation error";
        case 3: return "Invalid plan";
        default: return "Unknown FFTW error";
    }
}
