#pragma once
#include "../DeconvolutionConfig.h"
#include "../DeconvolutionProcessor.h" // to include rectangleshape

class DeconvolutionAlgorithm{
public:
    virtual void configure(const DeconvolutionConfig& config) = 0;
    virtual void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) = 0;
    void setBackend(std::shared_ptr<IDeconvolutionBackend> backend){this->backend = backend;}
    virtual std::unique_ptr<DeconvolutionAlgorithm> clone() const = 0;
   
    
protected:
    double complexDivisionEpsilon;
    std::shared_ptr<IDeconvolutionBackend> backend;
};