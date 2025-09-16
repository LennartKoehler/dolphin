#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    // Constructor that takes a backend parameter
    
    void deconvolve(const FFTWData& H, const FFTWData& g, FFTWData& f) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    int iterations;



};
