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

#include "dolphin/deconvolution/algorithms/ConvolutionAlgorithm.h"
#include <iostream>
#include <cassert>
#include <spdlog/spdlog.h>

void ConvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    // No specific parameters for basic convolution
}

void ConvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Convolution algorithm initialization");\
    
    initialized = true;
}

bool ConvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void ConvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    assert(backend && "No backend available for Convolution algorithm");\
    
    assert(initialized && "Convolution algorithm not initialized. Call init() first.");\

    backend->getDeconvManager().forwardFFT(g, f);
    backend->getDeconvManager().complexMultiplication(f, H, f);
    backend->getDeconvManager().backwardFFT(f, f);

    complex_t norm = { static_cast<real_t>(1.0 / g.size.getVolume()), 0.0};
    backend->getDeconvManager().scalarMultiplication(f, norm, f); // Add normalization
}

std::unique_ptr<DeconvolutionAlgorithm> ConvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<ConvolutionAlgorithm>();
    // Copy all relevant state
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t ConvolutionAlgorithm::getMemoryMultiplier() const {
    return 0; // No additional memory allocation needed
}