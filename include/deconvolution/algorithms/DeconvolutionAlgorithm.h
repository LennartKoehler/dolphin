#pragma once
#include "../DeconvolutionConfig.h"
#include "../DeconvolutionProcessor.h" // to include rectangleshape

class DeconvolutionAlgorithm{
public:
    virtual void configure(const DeconvolutionConfig& config) = 0;
    virtual void deconvolve(const FFTWData& H, const FFTWData& g, FFTWData& f) = 0;
    void setBackend(std::shared_ptr<IDeconvolutionBackend> backend){this->backend = backend;}
    
protected:
    double complexDivisionEpsilon;
    std::shared_ptr<IDeconvolutionBackend> backend;
};