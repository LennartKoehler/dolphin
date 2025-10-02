#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    // Constructor that takes a backend parameter
    
    void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) override;
    void configure(const DeconvolutionConfig& config) override;
    std::unique_ptr<DeconvolutionAlgorithm> clone() const override;
    size_t getMemoryMultiplier() const override;
private:
    int iterations;



};
