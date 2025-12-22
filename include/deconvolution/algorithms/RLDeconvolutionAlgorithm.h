#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    // Constructor that takes a backend parameter
    RLDeconvolutionAlgorithm() = default;
    ~RLDeconvolutionAlgorithm() = default;
    
    void configure(const DeconvolutionConfig& config) override;
    void init(const RectangleShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    size_t getMemoryMultiplier() const override;
private:
    int iterations;
    bool initialized = false;
    
    // Algorithm-specific data members for intermediate calculations
    ComplexData c;
    
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};
