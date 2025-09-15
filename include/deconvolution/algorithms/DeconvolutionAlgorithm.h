#pragma once
#include "../DeconvolutionConfig.h"
#include "../DeconvolutionProcessor.h" // to include rectangleshape

class DeconvolutionAlgorithm{
public:
    virtual void configure(const DeconvolutionConfig& config) = 0;
    virtual void deconvolve(const FFTWData& H, const FFTWData& g, FFTWData& f) = 0;

protected:
    double complexDivisionEpsilon;
};