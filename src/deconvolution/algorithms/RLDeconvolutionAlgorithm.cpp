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

#include "dolphin/deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>
#include <spdlog/spdlog.h>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
}

void RLDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy algorithm initialization");
    
    // Allocate memory for intermediate arrays
    c = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    
    initialized = true;
}

bool RLDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLDeconvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    assert(backend && "No backend available for Richardson-Lucy algorithm");
    
    assert(initialized && "Richardson-Lucy algorithm not initialized. Call init() first.");

    // Use pre-allocated memory for intermediate arrays
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");
    backend->getMemoryManager().memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {

        progressFunction(iterations);
        // a) First transformation: Fn = FFT(fn)
        backend->getDeconvManager().forwardFFT(f, f);

        // Fn\' = Fn * H
        backend->getDeconvManager().complexMultiplication(f, H, c);

        // fn\' = IFFT(Fn\') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);
        // backend->getDeconvManager().scalarMultiplication(c, 1.0 / g.size.getVolume(), c); // Add normalization


        // b) Calculation of the Correction Factor: c = g / fn\'
        backend->getDeconvManager().complexDivision(g, c, c, complexDivisionEpsilon);

        // // c) Second transformation: C = FFT(c)
        backend->getDeconvManager().forwardFFT(c, c);

        // // C\' = C * conj(H)
        backend->getDeconvManager().complexMultiplicationWithConjugate(c, H, c);

        // // c\' = IFFT(C\') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);
        // backend->getDeconvManager().scalarMultiplication(c, 1.0 / g.size.getVolume(), c); // Add normalization


        backend->getDeconvManager().backwardFFT(f, f);
        // backend->getDeconvManager().scalarMultiplication(f, 1.0 / g.size.getVolume(), f); // Add normalization

        backend->getDeconvManager().complexMultiplication(f, c, f);
        
    }
    // backend->getMemoryManager().freeMemoryOnDevice(c); // dont need because it is managed within complexdatas destructor
}



std::unique_ptr<DeconvolutionAlgorithm> RLDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 1; // Allocates 1 additional array of input size
}