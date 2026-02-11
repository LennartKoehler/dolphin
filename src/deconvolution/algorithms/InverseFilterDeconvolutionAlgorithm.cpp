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

#include "dolphin/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>
#include <spdlog/spdlog.h>

void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Set epsilon for stabilized division
    epsilon = config.epsilon;  // Assuming epsilon is in the config
}

void InverseFilterDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    if (!backend) {
        spdlog::error("No backend available for Inverse Filter algorithm initialization");
        return;
    }
    
    // No additional memory allocations needed for this simple algorithm
    // All operations are done in-place or using temporary variables from the backend
    initialized = true;
}

bool InverseFilterDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void InverseFilterDeconvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    if (!backend) {
        spdlog::error("No backend available for Inverse Filter algorithm");
        return;
    }
    
    if (!initialized) {
        spdlog::error("Inverse Filter algorithm not initialized. Call init() first.");
        return;
    }

    // Verify inputs are on device
    assert(backend->getMemoryManager().isOnDevice(H.data) && "PSF is not on device");
    assert(backend->getMemoryManager().isOnDevice(g.data) && "Input image is not on device");
    assert(backend->getMemoryManager().isOnDevice(f.data) && "Output buffer is not on device");



    // Forward FFT on image
    backend->getDeconvManager().forwardFFT(g, g);

    // Division in frequency domain: F = G / H (with stabilization)
    backend->getDeconvManager().complexDivision(g, H, f, epsilon);

    // Inverse FFT to get result
    backend->getDeconvManager().backwardFFT(f, f);

    // Optional: Apply normalization if needed

    complex_t norm = { static_cast<real_t>(1.0 / g.size.getVolume()), 0.0};
    backend->getDeconvManager().scalarMultiplication(f, norm, f); // Add normalization



}

std::unique_ptr<DeconvolutionAlgorithm> InverseFilterDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<InverseFilterDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->epsilon = this->epsilon;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t InverseFilterDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 1; // Allocates 1 additional array of input size
}
