#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RLADDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    RLADDeconvolutionAlgorithm() = default;
    virtual ~RLADDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const RectangleShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    size_t getMemoryMultiplier() const override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
    int iterations;
    int dampingDecrease; //0=exp, 1=lin
    double alpha;
    double beta;
    bool initialized = false;
    
    // Algorithm-specific data members for intermediate calculations
    ComplexData c;
};