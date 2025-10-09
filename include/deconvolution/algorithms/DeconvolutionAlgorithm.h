#pragma once
#include "../DeconvolutionConfig.h"
#include "complexType.h"
#include "deconvolution/IDeconvolutionBackend.h"

class DeconvolutionAlgorithm{
public:
    DeconvolutionAlgorithm() = default;
    virtual void configure(const DeconvolutionConfig& config) = 0;
    // it is assumed that the input of convolve is already located on the backend device
    virtual void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) = 0;
    void setBackend(std::shared_ptr<IDeconvolutionBackend> backend){this->backend = backend;}
    inline std::unique_ptr<DeconvolutionAlgorithm> clone() const{
        std::unique_ptr<DeconvolutionAlgorithm> clone = cloneSpecific();
        clone->setBackend(this->backend);
        return clone;
    }
    virtual size_t getMemoryMultiplier() const = 0;
    
protected:
    virtual std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const = 0;
    double complexDivisionEpsilon = 1e-9; // should be in backend ?
    std::shared_ptr<IDeconvolutionBackend> backend;
    
    // Friend declaration for DeconvolutionProcessor to access cloneSpecific()
    friend class DeconvolutionProcessor;
};