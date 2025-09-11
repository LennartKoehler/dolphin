#pragma once

#include "deconvolution/algorithms/BaseDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h"
#include <iostream>

class InverseFilterDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm, public BaseDeconvolutionAlgorithmCPU {
public:
    void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    void configure(const DeconvolutionConfig& config);

private:
    // Algorithm-specific implementation of virtual methods
    void configureAlgorithmSpecific(const DeconvolutionConfig& config) override;
    bool preprocessBackendSpecific(int channel_num, int psf_index) override;
    void algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    bool postprocessBackendSpecific(int channel_num, int psf_index) override;
    bool allocateBackendMemory(int channel_num) override;
    void deallocateBackendMemory(int channel_num) override;
    void cleanupBackendSpecific() override;
};

