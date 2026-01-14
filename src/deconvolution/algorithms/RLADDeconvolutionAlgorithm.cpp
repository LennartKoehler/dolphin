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

#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include <iostream>

#include <cassert>

void RLADDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
    dampingDecrease = 0; // Fixed to exponential decay as in original
    alpha = 0.9;         // Fixed as in original
    beta = 0.01;         // Fixed as in original
}

void RLADDeconvolutionAlgorithm::init(const RectangleShape& dataSize) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for RLAD algorithm initialization" << std::endl;
        return;
    }
    
    // Allocate memory for intermediate arrays
    c = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    
    initialized = true;
}

bool RLADDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLADDeconvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for RLAD algorithm" << std::endl;
        return;
    }
    
    if (!initialized) {
        std::cerr << "[ERROR] RLAD algorithm not initialized. Call init() first." << std::endl;
        return;
    }

    // Use pre-allocated memory for intermediate arrays
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");
    backend->getMemoryManager().memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {
        // Calculate damping factor
        double a;
        if (dampingDecrease == 0) {
            a = alpha * exp(-beta * n);
        } else {  // Linear decay
            a = alpha - beta * n;
        }


        // a) First transformation: Fn = FFT(fn)
        backend->getDeconvManager().forwardFFT(f, f);

        // Fn' = Fn * H
        backend->getDeconvManager().complexMultiplication(f, H, c);

        // fn' = IFFT(Fn') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);

        // b) Calculation of the Correction Factor: c = g / fn'
        backend->getDeconvManager().complexDivision(g, c, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        backend->getDeconvManager().forwardFFT(c, c);

        // C' = C * conj(H)
        backend->getDeconvManager().complexMultiplicationWithConjugate(c, H, c);

        // c' = IFFT(C') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);

        // d) Update the estimated image:
        backend->getDeconvManager().backwardFFT(f, f);

        // Apply adaptive damping: c = c * a
        backend->getDeconvManager().scalarMultiplication(c, a, c);

        // fn+1' = fn * c
        backend->getDeconvManager().complexMultiplication(f, c, f);
        
    }
    
}

std::unique_ptr<DeconvolutionAlgorithm> RLADDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLADDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->dampingDecrease = this->dampingDecrease;
    copy->alpha = this->alpha;
    copy->beta = this->beta;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLADDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 4; // Allocates 1 additional array of input size + 3 input copies
}
