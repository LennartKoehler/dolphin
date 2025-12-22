#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>

class InverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    InverseFilterDeconvolutionAlgorithm() = default;
    virtual ~InverseFilterDeconvolutionAlgorithm() = default;

    // Main algorithm interface
    void configure(const DeconvolutionConfig& config) override;
    void init(const RectangleShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    
    size_t getMemoryMultiplier() const override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
    double epsilon = 1e-6;  // Stabilization parameter for division
    bool initialized = false;
};

