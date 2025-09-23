#pragma once

#include "deconvolution/algorithms/BaseDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h"
#include <iostream>

class RLADDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm, public BaseDeconvolutionAlgorithmCPU {
public:
    void algorithm(Hyperstack& data, int channel_num, complex* H, complex* g, complex* f) override;
    void configure(const DeconvolutionConfig& config);

private:
    int iterations;
    int dampingDecrease; //0=exp, 1=lin
    double alpha;
    double beta;
    
    // Algorithm-specific implementation of virtual methods
    void configureAlgorithmSpecific(const DeconvolutionConfig& config) override;
    bool preprocessBackendSpecific(int channel_num, int psf_index) override;
    void algorithmBackendSpecific(int channel_num, complex* H, complex* g, complex* f) override;
    bool postprocessBackendSpecific(int channel_num, int psf_index) override;
    bool allocateBackendMemory(int channel_num) override;
    void deallocateBackendMemory(int channel_num) override;
    void cleanupBackendSpecific() override;
};