#pragma once
#include "../DeconvolutionConfig.h"
#include "../DeconvolutionProcessor.h" // to include rectangleshape

class DeconvolutionAlgorithm{
public:
    DeconvolutionAlgorithm() = default;
    virtual void configure(const DeconvolutionConfig& config) = 0;
    // it is assumed that the input of convolve is already located on the backend device
    virtual void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) = 0;
    void setBackend(std::shared_ptr<IDeconvolutionBackend> backend){this->backend = backend;}
    virtual std::unique_ptr<DeconvolutionAlgorithm> clone() const = 0;
   
    
protected:
    double complexDivisionEpsilon = 1e-6; // should be in backend ?
    std::shared_ptr<IDeconvolutionBackend> backend;
};