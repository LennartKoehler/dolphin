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

#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>

class InverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    InverseFilterDeconvolutionAlgorithm() = default;
    virtual ~InverseFilterDeconvolutionAlgorithm() = default;

    // Main algorithm interface
    void configure(const DeconvolutionConfig& config) override;
    void init(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    
    size_t getMemoryMultiplier() const override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
    double epsilon = 1e-6;  // Stabilization parameter for division
    bool initialized = false;
};

