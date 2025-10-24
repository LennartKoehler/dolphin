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

#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>

void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Set epsilon for stabilized division
    epsilon = config.epsilon;  // Assuming epsilon is in the config
}

void InverseFilterDeconvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Inverse Filter algorithm" << std::endl;
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
    // backend->getDeconvManager().scalarMultiplication(f, 1.0 / g.size.volume, f);


}

std::unique_ptr<DeconvolutionAlgorithm> InverseFilterDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<InverseFilterDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->epsilon = this->epsilon;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t InverseFilterDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 4; // Allocates 1 additional array of input size + 3 input copies
}
