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



void ConvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Test algorithm has no configuration parameters
}

void ConvolutionAlgorithm::init(const CuboidShape& dataSize) {
    // Test algorithm doesn't need any special initialization or memory allocation
    initialized = true;
}

bool ConvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void ConvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    backend->getDeconvManager().forwardFFT(g, f);
    backend->getDeconvManager().complexMultiplication(f, H, f);
    backend->getDeconvManager().backwardFFT(f, f);

    complex_t norm = { static_cast<real_t>(1.0 / g.size.getVolume()), 0.0};
    backend->getDeconvManager().scalarMultiplication(f, norm, f); // Add normalization
    
}

std::unique_ptr<DeconvolutionAlgorithm> ConvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<ConvolutionAlgorithm>();
    copy->initialized = false; // Clone needs to be re-initialized
    return copy;
}

size_t ConvolutionAlgorithm::getMemoryMultiplier() const {
    return 3; // No additional memory allocation + 3 input copies
}
