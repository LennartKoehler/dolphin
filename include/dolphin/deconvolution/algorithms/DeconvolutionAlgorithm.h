/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once
#include "../DeconvolutionConfig.h"
#include "dolphinbackend/ComplexData.h"
#include "backend/BackendFactory.h"

class DeconvolutionAlgorithm{
public:
    DeconvolutionAlgorithm() = default;
    virtual ~DeconvolutionAlgorithm() = default;
    virtual void configure(const DeconvolutionConfig& config) = 0;
    
    // Initialize algorithm-specific data allocations
    virtual void init(const RectangleShape& dataSize) = 0;
    virtual bool isInitialized() const = 0;
    
    // it is assumed that the input of convolve is already located on the backend device
    virtual void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) = 0;
    void setBackend(std::shared_ptr<IBackend> backend){this->backend = backend;}
    inline std::unique_ptr<DeconvolutionAlgorithm> clone() const{
        std::unique_ptr<DeconvolutionAlgorithm> clone = cloneSpecific();
        return clone;
    }
    virtual size_t getMemoryMultiplier() const = 0;
    
protected:
    virtual std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const = 0;
    double complexDivisionEpsilon = 1e-9; // should be in backend ?
    std::shared_ptr<IBackend> backend;
    

};