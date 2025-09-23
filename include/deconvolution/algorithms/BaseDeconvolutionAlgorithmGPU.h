#pragma once

#include "BaseDeconvolutionAlgorithmDerived.h"
#include <vector>
#include <memory>
#include <chrono>

#include <cuda_runtime.h>
#include <cufftw.h>
#include <CUBE.h>


typedef cufftComplex cufftComplex_t;
typedef cufftHandle cufftHandle_t;
typedef cufftResult cufftResult_t;
typedef cudaError_t cudaError_t_t;
typedef cudaStream_t cudaStream_t_t;
typedef cudaDeviceProp cudaDeviceProp_t;


/**
 * GPU-specific base class for deconvolution algorithms that implements 
 * all CUDA/CUFFT-based processing operations.
 * 
 * This class provides concrete implementations for all backend-specific
 * virtual methods in DeconvolutionProcessor, using CUDA and CUFFT
 * for efficient GPU-based FFT processing.
 */
class BaseDeconvolutionAlgorithmGPU : public DeconvolutionProcessor {
public:
    BaseDeconvolutionAlgorithmGPU();
    virtual ~BaseDeconvolutionAlgorithmGPU();

    // Override backend-specific virtual methods
    virtual bool preprocessBackendSpecific(int channel_num, int psf_index) override;
    virtual void algorithmBackendSpecific(int channel_num, complex* H, complex* g, complex* f) override;
    virtual bool postprocessBackendSpecific(int channel_num, int psf_index) override;
    virtual bool allocateBackendMemory(int channel_num) override;
    virtual void deallocateBackendMemory(int channel_num) override;
    virtual void cleanupBackendSpecific() override;
    virtual void configureAlgorithmSpecific(const DeconvolutionConfig& config) override;

    // GPU-specific public interface
    bool isGPUSupported() const;
    int getGPUDeviceCount() const;
    std::vector<int> getAvailableGPUs() const;
    bool setGPUDevice(int device_id);
    int getCurrentGPUDevice() const;
    
    // Performance monitoring
    double getLastGPURuntime() const;
    std::vector<double> getGPURuntimeHistory() const;
    void resetGPUStats();

protected:
    // Helper functions for CUFFT operations
    bool createCUFFTPlans();
    void destroyCUFFTPlans();
    bool executeForwardGPUFFT(cufftComplex_t* input, cufftComplex_t* output);
    bool executeBackwardGPUFFT(cufftComplex_t* input, cufftComplex_t* output);
    bool validateCUFFTPlan(cufftHandle_t plan);
    
    // GPU memory management
    bool allocateGPUArray(cufftComplex_t*& array, size_t size);
    void deallocateGPUArray(cufftComplex_t* array);
    bool allocateHostPinnedArray(complex*& array, size_t size);
    void deallocateHostPinnedArray(complex* array);
    
    // Data transfer utilities
    bool copyToGPU(cufftComplex_t* device_array, const complex* host_array, size_t size);
    bool copyFromGPU(complex* host_array, const cufftComplex_t* device_array, size_t size);

    // Helper methods for channel memory management
    void cleanupChannelMemory(int channel_num);
    

    // Performance tracking
    void startGPUTimer();
    void stopGPUTimer();
    double getGPUTimerDuration();
    
    // Diagnostics and monitoring
    void logGPUDeviceInfo(int device_id) const;
    void logGPUComputeCapability(int device_id) const;
    size_t getAvailableGPUMemory(int device_id) const;
    size_t getTotalGPUMemory(int device_id) const;
    void logMemoryUsage(const std::string& operation) const;
    
    // CUDA stream management
    cudaStream_t_t getStreamForChannel(int channel_num);
    void releaseStreamForChannel(int channel_num);
    
private:
    // CUFFT plan management
    cufftHandle_t m_forwardPlan;
    cufftHandle_t m_backwardPlan;
    
    // GPU memory management
    std::vector<cufftComplex_t*> m_allocatedGPUArrays;
    std::vector<complex*> m_allocatedPinnedArrays;
    std::vector<std::vector<cufftComplex_t*>> m_channelSpecificGPUMemory;

    // CUDA streams for asynchronous operations
    std::vector<cudaStream_t_t> m_gpuStreams;
    int m_maxGPUStreams;
    bool m_useAsyncTransfers;
    
    // GPU device management
    int m_currentGPUDevice;
    int m_preferredGPUDevice;
    std::vector<int> m_availableGPUDevices;
    bool m_gpuDeviceSelected;
    bool m_useMultiGPU;
    
    // CUBE context
    void* m_cubeContext;
    bool m_cubeInitialized;
    
    // Configuration flags
    bool m_cufftInitialized;
    bool m_cudaInitialized;
    bool m_usePinnedMemory;
    bool m_optimizePlans;
    bool m_enableErrorChecking;
    bool m_useCUBEKernels;
    
    // CUDA device properties
    cudaDeviceProp_t m_deviceProps;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point m_gpuStartTime;
    std::vector<double> m_gpuExecutionTimes;
    double m_lastGPURuntime;
    bool m_timingEnabled;
    
    // Memory tracking
    size_t m_allocatedGPUMemory;
    size_t m_peakGPUMemory;
    
    // Internal helper methods
    bool setupCUDAEnvironment();
    void initializeGPUDevices();
    bool selectOptimalGPU();
    void cleanupGPUResources();
    void movePSFstoGPU(std::unordered_map<PSFIndex, PSFfftw*>& psfMap);
    
    // Performance utilities
    void logPerformanceMetrics();

    
    // Configuration validation
    bool validateGPUConfig(const DeconvolutionConfig& config);
    void applyGPUSpecificSettings();
    

};