#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#include "UtlImage.h"
#include "UtlGrid.h"
#include "UtlFFT.h"
#include "backend/Exceptions.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <sstream>
#include <iomanip>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cufftw.h>
#include <CUBE.h>
#else
#include <fftw3.h>
#endif

using namespace std;

// Helper macro for CUDA error checking (legacy interface)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "[CUDA ERROR] " << #call << " failed at " << __FILE__ << ":" << __LINE__ << endl; \
            cerr << "[CUDA ERROR] " << cudaGetErrorString(err) << endl; \
            return false; \
        } \
    } while(0)

// Helper macro for CUFFT error checking (legacy interface)
#define CUFFT_CHECK(call) \
    do { \
        cufftResult_t err = call; \
        if (err != CUFFT_SUCCESS) { \
            cerr << "[CUFFT ERROR] " << #call << " failed at " << __FILE__ << ":" << __LINE__ << endl; \
            return false; \
        } \
    } while(0)

// Unified error checking macros (preferred interface)
#define CUDA_UNIFIED_CHECK(call, operation) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw dolphin::backend::BackendException( \
            std::string("CUDA error: ") + cudaGetErrorString(err), \
            "CUDA", \
            operation \
        ); \
    } \
}

#define CUFFT_UNIFIED_CHECK(call, operation) { \
    cufftResult_t res = call; \
    if (res != CUFFT_SUCCESS) { \
        throw dolphin::backend::BackendException( \
            "cuFFT error code: " + std::to_string(res), \
            "CUDA", \
            operation \
        ); \
    } \
}

BaseDeconvolutionAlgorithmGPU::BaseDeconvolutionAlgorithmGPU() 
    : m_forwardPlan(nullptr)
    , m_backwardPlan(nullptr)
    , m_currentGPUDevice(0)
    , m_preferredGPUDevice(0)
    , m_gpuDeviceSelected(false)
    , m_useMultiGPU(false)
    , m_cubeContext(nullptr)
    , m_cubeInitialized(false)
    , m_cufftInitialized(false)
    , m_cudaInitialized(false)
    , m_usePinnedMemory(false)
    , m_optimizePlans(false)
    , m_enableErrorChecking(true)
    , m_useCUBEKernels(false)
    , m_allocatedGPUMemory(0)
    , m_peakGPUMemory(0)
    , m_maxGPUStreams(8)
    , m_useAsyncTransfers(true)
    , m_timingEnabled(false)
    , m_lastGPURuntime(0.0) {
    
    cout << "[STATUS] BaseDeconvolutionAlgorithmGPU constructor" << endl;
    
    m_cudaInitialized = setupCUDAEnvironment();
    if (m_cudaInitialized) {
        initializeGPUDevices();
        selectOptimalGPU();
    } else {
        cerr << "[ERROR] CUDA environment setup failed" << endl;
    }
    
    // Initialize timer
    startGPUTimer();

}

BaseDeconvolutionAlgorithmGPU::~BaseDeconvolutionAlgorithmGPU() {
    cout << "[STATUS] BaseDeconvolutionAlgorithmGPU destructor" << endl;
    cleanupBackendSpecific();
}

// Implementation of pure virtual methods from DeconvolutionProcessor
bool BaseDeconvolutionAlgorithmGPU::preprocessBackendSpecific(int channel_num, int psf_index) {
    cout << "[STATUS] GPU preprocessing for channel " << channel_num << ", PSF " << psf_index << endl;
    
    if (!m_cudaInitialized) {
        cerr << "[ERROR] CUDA not initialized for preprocessing" << endl;
        return false;
    }
    
    try {
        // Allocate GPU memory for current channel
        if (!allocateBackendMemory(channel_num)) {
            cerr << "[ERROR] Failed to allocate GPU memory for channel " << channel_num << endl;
            return false;
        }
        
        // Create CUFFT plans for the current channel
        if (!createCUFFTPlans()) {
            cerr << "[ERROR] Failed to create CUFFT plans for channel " << channel_num << endl;
            return false;
        }
    }
}



void BaseDeconvolutionAlgorithmGPU::algorithmBackendSpecific(int channel_num, complex_t* H, complex_t* g, complex_t* f) {
    cout << "[STATUS] GPU algorithm execution for channel " << channel_num << endl;
    
    if (!m_cudaInitialized || !m_cufftInitialized) {
        cerr << "[ERROR] CUDA/CUFFT not initialized for algorithm execution" << endl;
        return;
    }
    
    try {
        startGPUTimer();
        
        // Convert input data to cufftComplex format
        cufftComplex_t* d_g = nullptr;
        cufftComplex_t* d_f = nullptr;
        
        if (!allocateGPUArray(d_g, cubeVolume)) {
            cerr << "[ERROR] Failed to allocate GPU memory for observation data" << endl;
            return;
        }
        
        if (!allocateGPUArray(d_f, cubeVolume)) {
            cerr << "[ERROR] Failed to allocate GPU memory for result data" << endl;
            deallocateGPUArray(d_g);
            return;
        }
        
        // Copy observation data to GPU
        if (!copyToGPU(d_g, g, cubeVolume)) {
            cerr << "[ERROR] Failed to copy observation data to GPU" << endl;
            deallocateGPUArray(d_g);
            deallocateGPUArray(d_f);
            return;
        }
        
        // Copy result template to GPU (initialized with zeros)
        cudaMemsetAsync(d_f, 0, cubeVolume * sizeof(cufftComplex_t), getStreamForChannel(channel_num));
        
        // Get the current stream for this channel
        cudaStream_t_t stream = getStreamForChannel(channel_num);
        
        // Execute forward FFT on observation data
        if (!executeForwardGPUFFT(g, d_g)) {
            cerr << "[ERROR] Failed to execute forward FFT on GPU" << endl;
            deallocateGPUArray(d_g);
            deallocateGPUArray(d_f);
            return;
        }
        

        
 
        
        if (m_useCUBEKernels && m_cubeInitialized) {
            if (!executeCUBEAlgorithm(channel_num, d_H, d_g, d_f)) {
                cerr << "[ERROR] CUBE algorithm execution failed" << endl;
                deallocateGPUArray(d_g);
                deallocateGPUArray(d_f);
                return;
            }
        } else {
            // Fallback to basic GPU operations
            // Apply PSF in frequency domain (convolution theorem)
            // This is a simplified version - actual implementation would be more complex_t
            cufftComplex_t* d_temp = nullptr;
            if (!allocateGPUArray(d_temp, cubeVolume)) {
                cerr << "[ERROR] Failed to allocate temporary GPU memory" << endl;
                deallocateGPUArray(d_g);
                deallocateGPUArray(d_f);
                return;
            }
            
            // Simple element-wise multiplication: temp = H * g (frequency domain convolution)
            for (size_t i = 0; i < cubeVolume; ++i) {
                d_temp[i].x = d_H[i].x * d_g[i].x - d_H[i].y * d_g[i].y;
                d_temp[i].y = d_H[i].x * d_g[i].y + d_H[i].y * d_g[i].x;
            }
            
            // Copy back to result
            cudaMemcpyAsync(d_f, d_temp, cubeVolume * sizeof(cufftComplex_t), cudaMemcpyDeviceToDevice, stream);
            
            deallocateGPUArray(d_temp);
        }
        
        // Execute inverse FFT to get spatial domain result
        if (!executeBackwardGPUFFT(d_f, reinterpret_cast<cufftComplex_t*>(f))) {
            cerr << "[ERROR] Failed to execute backward FFT on GPU" << endl;
            deallocateGPUArray(d_g);
            deallocateGPUArray(d_f);
            return;
        }
        
        stopGPUTimer();
        m_lastGPURuntime = getGPUTimerDuration();
        m_gpuExecutionTimes.push_back(m_lastGPURuntime);
        
        // Cleanup
        deallocateGPUArray(d_g);
        deallocateGPUArray(d_f);
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU algorithm execution: " << e.what() << endl;
    }
    

}

bool BaseDeconvolutionAlgorithmGPU::postprocessBackendSpecific(int channel_num, int psf_index) {
    cout << "[STATUS] GPU postprocessing for channel " << channel_num << ", PSF " << psf_index << endl;
    
    if (!m_cudaInitialized) {
        cerr << "[ERROR] CUDA not initialized for postprocessing" << endl;
        return false;
    }
    
    try {
        // Copy results back to host
        // The real result data is already copied in algorithmBackendSpecific
        
        // Apply any necessary GPU-based filtering or normalization
        // This would typically involve additional CUDA kernels
        
        // Cleanup channel-specific GPU resources
        cleanupChannelMemory(channel_num);
        
        // Log performance metrics
        if (m_timingEnabled) {
            logPerformanceMetrics();
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU postprocessing: " << e.what() << endl;
        return false;
    }
    

}

bool BaseDeconvolutionAlgorithmGPU::allocateBackendMemory(int channel_num) {
    cout << "[STATUS] Allocating GPU memory for channel " << channel_num << endl;
    
    if (!m_cudaInitialized) {
        cerr << "[ERROR] CUDA not initialized for memory allocation" << endl;
        return false;
    }
    
    try {
        // Get current stream for this channel
        cudaStream_t_t stream = getStreamForChannel(channel_num);
        
        // Check if channel-specific memory already exists
        if (channel_num >= m_channelSpecificGPUMemory.size()) {
            m_channelSpecificGPUMemory.resize(channel_num + 1);
            m_channelSpecificGPUMemory[channel_num].clear();
        }
        
        // Allocate main working memory arrays
        cufftComplex_t* d_work1 = nullptr;
        cufftComplex_t* d_work2 = nullptr;
        cufftComplex_t* d_work3 = nullptr;
        
        if (!allocateGPUArray(d_work1, cubeVolume)) {
            cerr << "[ERROR] Failed to allocate first working array" << endl;
            return false;
        }
        
        if (!allocateGPUArray(d_work2, cubeVolume)) {
            cerr << "[ERROR] Failed to allocate second working array" << endl;
            deallocateGPUArray(d_work1);
            return false;
        }
        
        if (!allocateGPUArray(d_work3, cubeVolume)) {
            cerr << "[ERROR] Failed to allocate third working array" << endl;
            deallocateGPUArray(d_work1);
            deallocateGPUArray(d_work2);
            return false;
        }
        
        // Store the allocated arrays
        m_channelSpecificGPUMemory[channel_num].push_back(d_work1);
        m_channelSpecificGPUMemory[channel_num].push_back(d_work2);
        m_channelSpecificGPUMemory[channel_num].push_back(d_work3);
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU memory allocation: " << e.what() << endl;
        return false;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::deallocateBackendMemory(int channel_num) {
    cout << "[STATUS] Deallocating GPU memory for channel " << channel_num << endl;
    
    if (!m_cudaInitialized) {
        cerr << "[ERROR] CUDA not initialized for memory deallocation" << endl;
        return;
    }
    
    try {
        cleanupChannelMemory(channel_num);
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU memory deallocation: " << e.what() << endl;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::cleanupBackendSpecific() {
    cout << "[STATUS] Cleaning up GPU-specific resources" << endl;
    
    try {
        // Destroy CUFFT plans
        destroyCUFFTPlans();
        
        // Cleanup channel-specific memory
        for (size_t i = 0; i < m_channelSpecificGPUMemory.size(); ++i) {
            cleanupChannelMemory(static_cast<int>(i));
        }
        m_channelSpecificGPUMemory.clear();
        
        // Cleanup PSF memory
        for (auto& d_psf : paddedHs) {
            if (d_psf) {
                deallocateGPUArray(d_psf);
                d_psf = nullptr;
            }
        }
        paddedHs.clear();
        
        // Cleanup GPU streams
        for (auto& stream : m_gpuStreams) {
            cudaStreamDestroy(stream);
        }
        m_gpuStreams.clear();
        
        // Cleanup CUBE context
        cleanupCUBEContext();
        
        // Reset CUDA environment
        m_cufftInitialized = false;
        m_cudaInitialized = false;
        m_cubeInitialized = false;
        
        // Print final memory statistics
        if (m_enableErrorChecking) {
            cout << "[STATS] Peak GPU memory usage: " << m_peakGPUMemory / (1024.0 * 1024.0) << " MB" << endl;
            cout << "[STATS] Total GPU execution time: " << accumulate(m_gpuExecutionTimes.begin(), m_gpuExecutionTimes.end(), 0.0) << " ms" << endl;
        }
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU cleanup: " << e.what() << endl;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    cout << "[STATUS] Configuring GPU-specific parameters" << endl;
    
    // Validate GPU configuration
    if (!validateGPUConfig(config)) {
        cerr << "[ERROR] Invalid GPU configuration" << endl;
        return;
    }
    
    // Extract GPU-specific parameters
    // m_usePinnedMemory = config.usePinnedMemory;
    // m_optimizePlans = config.optimizePlans;
    // m_enableErrorChecking = config.enableErrorChecking;
    // m_useCUBEKernels = config.useCUBEKernels;
    // m_preferredGPUDevice = config.preferredGPUDevice;
    
    // Apply GPU-specific settings
    applyGPUSpecificSettings();
    
    // Print configuration summary
    cout << "[CONFIG] GPU device: " << m_currentGPUDevice << endl;
    cout << "[CONFIG] Use pinned memory: " << (m_usePinnedMemory ? "true" : "false") << endl;
    cout << "[CONFIG] Optimize plans: " << (m_optimizePlans ? "true" : "false") << endl;
    cout << "[CONFIG] Use CUBE kernels: " << (m_useCUBEKernels ? "true" : "false") << endl;
    cout << "[CONFIG] Enable error checking: " << (m_enableErrorChecking ? "true" : "false") << endl;
    

}

// Public GPU interface methods
bool BaseDeconvolutionAlgorithmGPU::isGPUSupported() const {
#ifdef CUDA_AVAILABLE
    return m_cudaInitialized && m_cufftInitialized;
#else
    return false;
#endif
}

int BaseDeconvolutionAlgorithmGPU::getGPUDeviceCount() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        cerr << "[WARNING] Failed to get device count: " << cudaGetErrorString(err) << endl;
        return 0;
    }
    return device_count;

}

std::vector<int> BaseDeconvolutionAlgorithmGPU::getAvailableGPUs() const {
    std::vector<int> available_devices;
    if (m_cudaInitialized) {
        int device_count = getGPUDeviceCount();
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaError_t err = cudaGetDeviceProperties(&prop, i);
            if (err == cudaSuccess) {
                available_devices.push_back(i);
            }
        }
    }

}

bool BaseDeconvolutionAlgorithmGPU::setGPUDevice(int device_id) {
    if (!m_cudaInitialized) {
        cerr << "[ERROR] CUDA not initialized for device selection" << endl;
        return false;
    }
    
    int device_count = getGPUDeviceCount();
    if (device_id < 0 || device_id >= device_count) {
        cerr << "[ERROR] Invalid device ID: " << device_id << " (valid range: 0-" << device_count-1 << ")" << endl;
        return false;
    }
    
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        cerr << "[ERROR] Failed to set device " << device_id << ": " << cudaGetErrorString(err) << endl;
        return false;
    }
    
    // Get device properties
    err = cudaGetDeviceProperties(&m_deviceProps, device_id);
    if (err != cudaSuccess) {
        cerr << "[ERROR] Failed to get device properties: " << cudaGetErrorString(err) << endl;
        return false;
    }
    
    m_currentGPUDevice = device_id;
    m_gpuDeviceSelected = true;
    
    cout << "[INFO] Selected GPU device " << device_id << ": " << m_deviceProps.name << endl;
    
    return true;
    

}

int BaseDeconvolutionAlgorithmGPU::getCurrentGPUDevice() const {
    return m_currentGPUDevice;

}

double BaseDeconvolutionAlgorithmGPU::getLastGPURuntime() const {
    return m_lastGPURuntime;
}

std::vector<double> BaseDeconvolutionAlgorithmGPU::getGPURuntimeHistory() const {
    return m_gpuExecutionTimes;
}

void BaseDeconvolutionAlgorithmGPU::resetGPUStats() {
    m_gpuExecutionTimes.clear();
    m_lastGPURuntime = 0.0;
    startGPUTimer();
}

// Protected helper method implementations
bool BaseDeconvolutionAlgorithmGPU::createCUFFTPlans() {
    if (!m_cufftInitialized) {
        cufftResult_t result = cufftInit();
        if (result != CUFFT_SUCCESS) {
            cerr << "[ERROR] Failed to initialize CUFFT" << endl;
            return false;
        }
        m_cufftInitialized = true;
    }
    
    try {
        if (m_forwardPlan != nullptr) {
            cufftDestroy(m_forwardPlan);
            m_forwardPlan = nullptr;
        }
        
        if (m_backwardPlan != nullptr) {
            cufftDestroy(m_backwardPlan);
            m_backwardPlan = nullptr;
        }
        
        // Create forward 3D CUFFT plan
        cufftResult_t result = cufftPlan3d(&m_forwardPlan, cubeDepth, cubeHeight, cubeWidth, CUFFT_C2C);
        if (result != CUFFT_SUCCESS) {
            cerr << "[ERROR] Failed to create forward CUFFT plan" << endl;
            return false;
        }
        
        // Create backward 3D CUFFT plan
        result = cufftPlan3d(&m_backwardPlan, cubeDepth, cubeHeight, cubeWidth, CUFFT_C2C);
        if (result != CUFFT_SUCCESS) {
            cerr << "[ERROR] Failed to create backward CUFFT plan" << endl;
            return false;
        }
        
        // Optimize plans if requested
        if (m_optimizePlans) {
            cufftSetAutoAllocation(m_forwardPlan, 0);
            cufftSetAutoAllocation(m_backwardPlan, 0);
        }
        
        cout << "[STATUS] CUFFT plans created (Forward: " << m_forwardPlan << ", Backward: " << m_backwardPlan << ")" << endl;
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception creating CUFFT plans: " << e.what() << endl;
        return false;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::destroyCUFFTPlans() {
    if (m_forwardPlan != nullptr) {
        cufftDestroy(m_forwardPlan);
        m_forwardPlan = nullptr;
    }
    
    if (m_backwardPlan != nullptr) {
        cufftDestroy(m_backwardPlan);
        m_backwardPlan = nullptr;
    }
    
    if (m_cufftInitialized) {
        cufftExit();
        m_cufftInitialized = false;
    }
}

bool BaseDeconvolutionAlgorithmGPU::executeForwardGPUFFT(cufftComplex_t* input, cufftComplex_t* output) {
    if (!m_cufftInitialized || m_forwardPlan == nullptr) {
        cerr << "[ERROR] CUFFT not initialized or no forward plan" << endl;
        return false;
    }
    
    try {
        // Note: CUFFT expects cufftComplex, but our generic class uses cufftComplex_t
        // In practice, we'd need to handle this conversion
        cufftComplex* d_input = reinterpret_cast<cufftComplex*>(input);
        cufftComplex* d_output = reinterpret_cast<cufftComplex*>(output);
        
        cufftResult_t result = cufftExecC2C(m_forwardPlan, d_input, d_output, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            cerr << "[ERROR] Forward FFT execution failed" << endl;
            return false;
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in forward FFT: " << e.what() << endl;
        return false;
    }
    

}

bool BaseDeconvolutionAlgorithmGPU::executeBackwardGPUFFT(cufftComplex_t* input, cufftComplex_t* output) {
    if (!m_cufftInitialized || m_backwardPlan == nullptr) {
        cerr << "[ERROR] CUFFT not initialized or no backward plan" << endl;
        return false;
    }
    
    try {
        // Note: CUFFT expects cufftComplex, but our generic class uses cufftComplex_t
        // In practice, we'd need to handle this conversion
        cufftComplex* d_input = reinterpret_cast<cufftComplex*>(input);
        cufftComplex* d_output = reinterpret_cast<cufftComplex*>(output);
        
        cufftResult_t result = cufftExecC2C(m_backwardPlan, d_input, d_output, CUFFT_BACKWARD);
        if (result != CUFFT_SUCCESS) {
            cerr << "[ERROR] Backward FFT execution failed" << endl;
            return false;
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in backward FFT: " << e.what() << endl;
        return false;
    }
    

}

bool BaseDeconvolutionAlgorithmGPU::validateCUFFTPlan(cufftHandle_t plan) {
    if (plan == nullptr) {
        return false;
    }
    
    cufftResult_t result = cufftValidatePlan(plan);
    return (result == CUFFT_SUCCESS);
    

}

bool BaseDeconvolutionAlgorithmGPU::allocateGPUArray(cufftComplex_t*& array, size_t size) {
    if (size == 0) {
        return false;
    }
    
    try {
        cudaError_t err = cudaMalloc(&array, size * sizeof(cufftComplex_t));
        if (err != cudaSuccess) {
            cerr << "[CUDA ERROR] Failed to allocate GPU memory: " << cudaGetErrorString(err) << endl;
            return false;
        }
        
        m_allocatedGPUArrays.push_back(array);
        m_allocatedGPUMemory += size * sizeof(cufftComplex_t);
        
        if (m_allocatedGPUMemory > m_peakGPUMemory) {
            m_peakGPUMemory = m_allocatedGPUMemory;
        }
        
        if (m_enableErrorChecking) {
            logMemoryUsage("GPU allocation");
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU array allocation: " << e.what() << endl;
        return false;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::deallocateGPUArray(cufftComplex_t* array) {
    if (array != nullptr) {
        cudaError_t err = cudaFree(array);
        if (err != cudaSuccess && m_enableErrorChecking) {
            cerr << "[WARNING] Failed to free GPU memory: " << cudaGetErrorString(err) << endl;
        }
        
        // Remove from allocated list and update memory tracking
        auto it = std::find(m_allocatedGPUArrays.begin(), m_allocatedGPUArrays.end(), array);
        if (it != m_allocatedGPUArrays.end()) {
            size_t size = cubeVolume; // Approximate size calculation
            m_allocatedGPUMemory -= size * sizeof(cufftComplex_t);
            m_allocatedGPUArrays.erase(it);
        }
        
        array = nullptr;
    }

}

// Continue with other helper method implementations...
bool BaseDeconvolutionAlgorithmGPU::allocateHostPinnedArray(complex_t*& array, size_t size) {
    if (!m_usePinnedMemory || size == 0) {
        return false;
    }
    
    try {
        cudaError_t err = cudaHostAlloc(&array, size * sizeof(complex_t), cudaHostAllocPortable);
        if (err != cudaSuccess) {
            cerr << "[CUDA ERROR] Failed to allocate pinned host memory: " << cudaGetErrorString(err) << endl;
            return false;
        }
        
        m_allocatedPinnedArrays.push_back(array);
        
        if (m_enableErrorChecking) {
            logMemoryUsage("Pinned host allocation");
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in pinned array allocation: " << e.what() << endl;
        return false;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::deallocateHostPinnedArray(complex_t* array) {
    if (array != nullptr && m_usePinnedMemory) {
        cudaError_t err = cudaFreeHost(array);
        if (err != cudaSuccess && m_enableErrorChecking) {
            cerr << "[WARNING] Failed to free pinned host memory: " << cudaGetErrorString(err) << endl;
        }
        
        auto it = std::find(m_allocatedPinnedArrays.begin(), m_allocatedPinnedArrays.end(), array);
        if (it != m_allocatedPinnedArrays.end()) {
            m_allocatedPinnedArrays.erase(it);
        }
        
        array = nullptr;
    }
}

bool BaseDeconvolutionAlgorithmGPU::copyToGPU(cufftComplex_t* device_array, const complex_t* host_array, size_t size) {
    if (!device_array || !host_array || size == 0) {
        return false;
    }
    
    try {
        cudaError_t err = cudaMemcpy(device_array, host_array, size * sizeof(complex_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cerr << "[CUDA ERROR] Failed to copy data to GPU: " << cudaGetErrorString(err) << endl;
            return false;
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU copy: " << e.what() << endl;
        return false;
    }
    

}

bool BaseDeconvolutionAlgorithmGPU::copyFromGPU(complex_t* host_array, const cufftComplex_t* device_array, size_t size) {
    if (!host_array || !device_array || size == 0) {
        return false;
    }
    
    try {
        cudaError_t err = cudaMemcpy(host_array, device_array, size * sizeof(cufftComplex_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cerr << "[CUDA ERROR] Failed to copy data from GPU: " << cudaGetErrorString(err) << endl;
            return false;
        }
        
        // Convert cufftComplex to fftw_complex format
        for (size_t i = 0; i < size; ++i) {
            host_array[i][0] = device_array[i].x;
            host_array[i][1] = device_array[i].y;
        }
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception from GPU copy: " << e.what() << endl;
        return false;
    }
    

}

cudaStream_t_t BaseDeconvolutionAlgorithmGPU::getStreamForChannel(int channel_num) {
    // Ensure we have enough streams
    while (channel_num >= m_gpuStreams.size()) {
        cudaStream_t_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            cerr << "[WARNING] Failed to create stream, using default stream" << endl;
            return 0;
        }
        m_gpuStreams.push_back(stream);
    }
    
    return m_gpuStreams[channel_num];

}

void BaseDeconvolutionAlgorithmGPU::releaseStreamForChannel(int channel_num) {
    if (channel_num >= 0 && channel_num < m_gpuStreams.size()) {
        cudaStreamDestroy(m_gpuStreams[channel_num]);
        m_gpuStreams[channel_num] = nullptr;
    }
}

void BaseDeconvolutionAlgorithmGPU::startGPUTimer() {
    m_gpuStartTime = std::chrono::high_resolution_clock::now();

}

void BaseDeconvolutionAlgorithmGPU::stopGPUTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_gpuStartTime);
    m_lastGPURuntime = duration.count() / 1000.0; // Convert to milliseconds

}

double BaseDeconvolutionAlgorithmGPU::getGPUTimerDuration() {
    return m_lastGPURuntime;
}

void BaseDeconvolutionAlgorithmGPU::logMemoryUsage(const std::string& operation) const {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err == cudaSuccess) {
        size_t used_mem = total_mem - free_mem;
        double used_mb = used_mem / (1024.0 * 1024.0);
        double total_mb = total_mem / (1024.0 * 1024.0);
        double free_mb = free_mem / (1024.0 * 1024.0);
        
        cout << "[MEMORY] " << operation << " - Used: " << used_mb << " MB, Free: " << free_mb 
             << " MB, Total: " << total_mb << " MB" << endl;
    } else if (m_enableErrorChecking) {
        cerr << "[WARNING] Failed to get memory info: " << cudaGetErrorString(err) << endl;
    }

}

// Private helper method implementations
bool BaseDeconvolutionAlgorithmGPU::setupCUDAEnvironment() {
    try {
        // Initialize CUDA
        cudaError_t err = cudaSetDevice(0); // Default to device 0
        if (err != cudaSuccess && err != cudaErrorAlreadyInUse) {
            cerr << "[ERROR] Failed to set CUDA device: " << cudaGetErrorString(err) << endl;
            return false;
        }
        
        // Initialize CUFFT
        cufftResult_t cufft_err = cufftInit();
        if (cufft_err != CUFFT_SUCCESS) {
            cerr << "[ERROR] Failed to initialize CUFFT" << endl;
            return false;
        }
        
        cout << "[STATUS] CUDA environment initialized successfully" << endl;
        
        return true;
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in CUDA environment setup: " << e.what() << endl;
        return false;
    }
    

}

void BaseDeconvolutionAlgorithmGPU::initializeGPUDevices() {
    int device_count = getGPUDeviceCount();
    cout << "[INFO] Found " << device_count << " GPU device(s)" << endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        
        if (err == cudaSuccess) {
            m_availableGPUDevices.push_back(i);
            cout << "[GPU " << i << "] " << prop.name << " - Compute capability: " << prop.major << "." << prop.minor
                 << ", Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0) << " MB" << endl;
        }
    }

}

void BaseDeconvolutionAlgorithmGPU::cleanupChannelMemory(int channel_num) {
    if (channel_num >= 0 && channel_num < m_channelSpecificGPUMemory.size()) {
        for (auto& ptr : m_channelSpecificGPUMemory[channel_num]) {
            if (ptr) {
                deallocateGPUArray(ptr);
            }
        }
        m_channelSpecificGPUMemory[channel_num].clear();
    }
}

void BaseDeconvolutionAlgorithmGPU::cleanupGPUResources() {
    // Clean up all allocated GPU arrays
    for (auto& ptr : m_allocatedGPUArrays) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    m_allocatedGPUArrays.clear();
    
    // Clean up pinned memory
    for (auto& ptr : m_allocatedPinnedArrays) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
    m_allocatedPinnedArrays.clear();
    
    // Reset memory tracking
    m_allocatedGPUMemory = 0;
    m_peakGPUMemory = 0;
    
    // Clean up streams
    for (auto& stream : m_gpuStreams) {
        cudaStreamDestroy(stream);
    }
    m_gpuStreams.clear();
}

bool BaseDeconvolutionAlgorithmGPU::validateGPUConfig(const DeconvolutionConfig& config) {
    // Basic validation logic
    if (config.preferredGPUDevice >= 0) {
        int device_count = getGPUDeviceCount();
        if (config.preferredGPUDevice >= device_count) {
            cerr << "[WARNING] preferredGPUDevice out of range, using auto-selection" << endl;
            return false; // Will cause auto-selection in applyGPUSpecificSettings
        }
    }
    
    return true;
    

}

void BaseDeconvolutionAlgorithmGPU::applyGPUSpecificSettings() {
    if (m_availableGPUDevices.empty()) {
        cerr << "[WARNING] No GPU devices available" << endl;
        return;
    }
    
    // Use preferred device if valid, otherwise select best available
    int target_device = m_preferredGPUDevice;
    if (target_device < 0 || target_device >= m_availableGPUDevices.size()) {
        // Auto-select device 0 as default
        target_device = 0;
        if (m_availableGPUDevices.size() > 0) {
            target_device = m_availableGPUDevices[0];
        }
    }
    
    setGPUDevice(target_device);
    
    // Setup pinned memory if requested
    if (m_usePinnedMemory) {
        setupGPUPinnedMemory();
    }
    
    // Setup streams for async operations
    if (m_useAsyncTransfers) {
        setupGPUStreams();
    }
    
    // Initialize CUBE if requested and available
    if (m_useCUBEKernels) {
        initializeCUBE();
    }

}

bool BaseDeconvolutionAlgorithmGPU::selectOptimalGPU() {
    if (m_availableGPUDevices.empty()) {
        cerr << "[ERROR] No GPU devices available for selection" << endl;
        return false;
    }
    
    // Simple selection: use the first available GPU
    int selected_device = m_availableGPUDevices[0];
    
    if (!setGPUDevice(selected_device)) {
        cerr << "[ERROR] Failed to select GPU device " << selected_device << endl;
        return false;
    }
    
    // Log GPU device info
    logGPUDeviceInfo(selected_device);
    
    return true;
    

}

size_t BaseDeconvolutionAlgorithmGPU::getAvailableGPUMemory(int device_id) const {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err == cudaSuccess) {
        size_t free_mem, total_mem;
        err = cudaMemGetInfo(&free_mem, &total_mem);
        
        if (err == cudaSuccess) {
            return free_mem;
        }
    }
    
    return 0;

}

size_t BaseDeconvolutionAlgorithmGPU::getTotalGPUMemory(int device_id) const {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err == cudaSuccess) {
        return prop.totalGlobalMem;
    }
    
    return 0;

}

void BaseDeconvolutionAlgorithmGPU::logGPUDeviceInfo(int device_id) const {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err == cudaSuccess) {
        cout << "[GPU_INFO] Device " << device_id << ": " << prop.name << endl;
        cout << "[GPU_INFO] Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "[GPU_INFO] Total memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0) << " MB" << endl;
        cout << "[GPU_INFO] Clock rate: " << (prop.clockRate / 1000.0) << " MHz" << endl;
        cout << "[GPU_INFO] Multiprocessors: " << prop.multiProcessorCount << endl;
    } else {
        cerr << "[ERROR] Failed to get device properties: " << cudaGetErrorString(err) << endl;
    }

}

void BaseDeconvolutionAlgorithmGPU::logPerformanceMetrics() {
    if (m_gpuExecutionTimes.empty()) {
        cout << "[PERF] No GPU performance data available" << endl;
        return;
    }
    
    double total_time = accumulate(m_gpuExecutionTimes.begin(), m_gpuExecutionTimes.end(), 0.0);
    double avg_time = total_time / m_gpuExecutionTimes.size();
    double max_time = *max_element(m_gpuExecutionTimes.begin(), m_gpuExecutionTimes.end());
    double min_time = *min_element(m_gpuExecutionTimes.begin(), m_gpuExecutionTimes.end());
    
    cout << "[PERF] GPU Performance Metrics:" << endl;
    cout << "[PERF]   Total executions: " << m_gpuExecutionTimes.size() << endl;
    cout << "[PERF]   Total time: " << total_time << " ms" << endl;
    cout << "[PERF]   Average time: " << avg_time << " ms" << endl;
    cout << "[PERF]   Min time: " << min_time << " ms" << endl;
    cout << "[PERF]   Max time: " << max_time << " ms" << endl;
    cout << "[PERF]   Peak memory usage: " << (m_peakGPUMemory / 1024.0 / 1024.0) << " MB" << endl;
}