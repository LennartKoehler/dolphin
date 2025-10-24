#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    // Constructor that takes a backend parameter
    
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    void configure(const DeconvolutionConfig& config) override;
    size_t getMemoryMultiplier() const override;
private:
    int iterations;
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};
