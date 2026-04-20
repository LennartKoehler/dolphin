// This file contains the implementation of the Richardson-Lucy deconvolution algorithm.
// Copyright by Lennart Koehler

// Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
// https://www.leibniz-hki.de/en/applied-systems-biology.html
// HKI-Center for Systems Biology of Infection
// Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
// Adolf-Reichwein-Straße 23, 07745 Jena, Germany
//
// The project code is licensed under the MIT license.
// See the LICENSE file provided with the code for the full license.
// */

#include "dolphin/deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include <cassert>
#include <spdlog/spdlog.h>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
}

void RLDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy algorithm initialization");

    // Allocate memory for intermediate arrays
    c_complex = backend->getMemoryManager().allocateMemoryOnDeviceComplex(dataSize);
    f_complex = backend->getMemoryManager().allocateMemoryOnDeviceComplex(dataSize);
    initialized = true;
}

bool RLDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f) {

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IDeconvolutionBackend& deconvolution = backend->getDeconvManager();

    assert(backend && "No backend available for Richardson-Lucy algorithm");

    assert(initialized && "Richardson-Lucy algorithm not initialized. Call init() first.");

    // Use pre-allocated memory for intermediate arrays
    assert(memory.isOnDevice(f.getData()) && "PSF is not on device");

    memory.memCopy(g, f);
    RealView c_real = memory.reinterpret(c_complex);


    for (int n = 0; n < iterations; ++n) {

        // a) First transformation: Fn = FFT(fn)
        deconvolution.forwardFFT(f, f_complex);

        // Fn\' = Fn * H
        deconvolution.complexMultiplication(f_complex, H, c_complex);

        // fn\' = IFFT(Fn\') + NORMALIZE
        deconvolution.backwardFFT(c_complex, c_real);
        // deconvolution.scalarMultiplication(c, 1.0 / g.size.getVolume(), c); // Add normalization

        // b) Calculation of the Correction Factor: c = g / fn\'
        deconvolution.division(g, c_real, c_real, complexDivisionEpsilon);

        // // c) Second transformation: C = FFT(c)
        deconvolution.forwardFFT(c_real, c_complex);

        // // C\' = C * conj(H)
        deconvolution.complexMultiplicationWithConjugate(c_complex, H, c_complex);

        // // c\' = IFFT(C\') + NORMALIZE
        deconvolution.backwardFFT(c_complex, c_real);

        // deconvolution.backwardFFT(f_complex, f);

        deconvolution.multiplication(f, c_real, f);

        backend->sync();
        progressFunction(iterations);
    }
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
