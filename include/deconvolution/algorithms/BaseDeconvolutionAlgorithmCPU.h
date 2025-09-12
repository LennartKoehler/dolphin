#pragma once

#include "BaseDeconvolutionAlgorithmDerived.h"
#include <fftw3.h>
#include <vector>
#include <memory>

/**
 * CPU-specific base class for deconvolution algorithms that implements 
 * all FFTW-based processing operations.
 * 
 * This class provides concrete implementations for all backend-specific
 * virtual methods in DeconvolutionProcessor, using FFTW for
 * efficient CPU-based FFT processing.
 */
class BaseDeconvolutionAlgorithmCPU : public DeconvolutionProcessor {
public:
    BaseDeconvolutionAlgorithmCPU();
    virtual ~BaseDeconvolutionAlgorithmCPU();

    // Override backend-specific virtual methods
    virtual bool preprocessBackendSpecific(int channel_num, int psf_index) override;
    virtual void algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    virtual bool postprocessBackendSpecific(int channel_num, int psf_index) override;
    virtual bool allocateBackendMemory(int channel_num) override;
    virtual void deallocateBackendMemory(int channel_num) override;
    virtual void cleanupBackendSpecific() override;
    virtual void configureAlgorithmSpecific(const DeconvolutionConfig& config) override;

protected:
    // Helper functions for FFTW operations
    bool createFFTWPlans();
    void destroyFFTWPlans();
    bool executeForwardFFT(fftw_complex* input, fftw_complex* output);
    bool executeBackwardFFT(fftw_complex* input, fftw_complex* output);
    bool validateFFTWPlan(fftw_plan plan);
    
    // CPU memory management
    bool allocateCPUArray(fftw_complex*& array, size_t size);
    void deallocateCPUArray(fftw_complex* array);
    bool manageChannelSpecificMemory(int channel_num);
    
    // Data transformation utilities
    bool validateComplexArray(fftw_complex* array, size_t size, const std::string& array_name);
    bool normalizeComplexArray(fftw_complex* array, size_t size, double epsilon = 1e-12);
    bool copyComplexArray(const fftw_complex* source, fftw_complex* destination, size_t size);
    
    // Error handling and diagnostics
    void logFFTWError(fftw_plan plan, const std::string& operation);
    bool checkMemoryCriticalSystem() const;
    void logMemoryUsage(const std::string& operation) const;
    
    // Configuration and optimization
    void optimizeFFTWPlans();
    void setupThreadedFFTW();
    
private:
    // FFTW plan management
    fftw_plan m_forwardPlan;
    fftw_plan m_backwardPlan;


    fftw_complex *fftwPlanMem = nullptr;
    // Memory management
    std::vector<fftw_complex*> m_allocatedArrays;
    std::vector<std::vector<fftw_complex*>> m_channelSpecificMemory;
    
    // Configuration flags
    bool m_fftwInitialized;
    bool m_optimizePlans;
    int m_fftwThreads;
    
    // Performance tracking
    std::vector<double> m_executionTimes;
    
    // Internal helper methods
    bool setupFFTWThreadEnvironment();
    bool validateMemoryAllocation(size_t required_size);
    void cleanupChannelMemory(int channel_num);
    void logPerformanceMetrics();
    
    // Error handling utilities
    bool handleFFTWError(fftw_plan plan, const std::string& operation);
    std::string getFFTWErrorString(int error_code) const;
};