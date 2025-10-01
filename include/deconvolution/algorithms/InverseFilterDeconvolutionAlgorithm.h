#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>

class InverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    InverseFilterDeconvolutionAlgorithm() = default;
    virtual ~InverseFilterDeconvolutionAlgorithm() = default;

    // Main algorithm interface
    void configure(const DeconvolutionConfig& config) override;
    void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) override;
    
    // Clone method for thread safety
    std::unique_ptr<DeconvolutionAlgorithm> clone() const override;

private:
    double epsilon = 1e-6;  // Stabilization parameter for division
};

