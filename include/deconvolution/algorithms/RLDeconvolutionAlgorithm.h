#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    // Constructor that takes a backend parameter
    explicit RLDeconvolutionAlgorithm(std::shared_ptr<IDeconvolutionBackend> backend);
    
    void deconvolve(const FFTWData& H, const FFTWData& g, FFTWData& f) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    int iterations;
    std::shared_ptr<IDeconvolutionBackend> backend;  // Backend pointer for backend-agnostic operations


};
