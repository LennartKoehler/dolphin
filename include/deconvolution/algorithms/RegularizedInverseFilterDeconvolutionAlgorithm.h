#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RegularizedInverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) override;
    void configure(const DeconvolutionConfig& config) override;
    size_t getMemoryMultiplier() const override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
    double lambda;
};
