#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>

class RLTVDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    RLTVDeconvolutionAlgorithm() = default;
    virtual ~RLTVDeconvolutionAlgorithm() = default;

    // Main algorithm interface
    void configure(const DeconvolutionConfig& config) override;
    void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) override;
    
    // Clone method for thread safety
    std::unique_ptr<DeconvolutionAlgorithm> clone() const override;

private:
    int iterations = 10;        // Number of RL iterations
    double lambda = 0.01;       // TV regularization parameter
    double complexDivisionEpsilon = 1e-6;  // Stabilization for division
};