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

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class RegularizedInverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    RegularizedInverseFilterDeconvolutionAlgorithm() = default;
    virtual ~RegularizedInverseFilterDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const RectangleShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    size_t getMemoryMultiplier() const override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
    double lambda;
    bool initialized = false;
    
    // Algorithm-specific data members for intermediate calculations
    ComplexData H2;
    ComplexData L;
    ComplexData L2;
    ComplexData FA;
    ComplexData FP;
};
