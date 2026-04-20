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

#include "dolphin/deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>
#include <spdlog/spdlog.h>

void RLADDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
    dampingDecrease = 0; // Fixed to exponential decay as in original
    alpha = 0.9;         // Fixed as in original
    beta = 0.01;         // Fixed as in original
}

void RLADDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy with Adaptive Damping algorithm initialization");\
    
    // Allocate memory for intermediate arrays
    c = std::move(backend->getMemoryManager().allocateMemoryOnDeviceReal(dataSize));
    c_complex = std::move(backend->getMemoryManager().allocateMemoryOnDeviceComplex(dataSize));
    
    initialized = true;
}

bool RLADDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLADDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f) {
    assert(backend && "No backend available for Richardson-Lucy with Adaptive Damping algorithm");\
    
    assert(initialized && "Richardson-Lucy with Adaptive Damping algorithm not initialized. Call init() first.");\
    
    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IDeconvolutionBackend& deconvolution = backend->getDeconvManager();

    // Use pre-allocated memory for intermediate arrays
    assert(memory.isOnDevice(f.getData()) && "PSF is not on device");
    memory.memCopy(g, f);

    // Allocate temporary complex buffer for f in frequency domain
    ComplexData f_complex = memory.allocateMemoryOnDeviceComplex(f.getSize());

    for (int n = 0; n < iterations; ++n) {
        // Calculate damping factor
        double a;
        if (dampingDecrease == 0) {
            a = alpha * exp(-beta * n);
        } else {  // Linear decay
            a = alpha - beta * n;
        }

        // a) First transformation: Fn = FFT(fn)
        deconvolution.forwardFFT(f, f_complex);

        // Fn' = Fn * H
        deconvolution.complexMultiplication(f_complex, H, c_complex);

        // fn' = IFFT(Fn')
        deconvolution.backwardFFT(c_complex, c);

        // b) Calculation of the Correction Factor: c = g / fn'
        deconvolution.division(g, c, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        deconvolution.forwardFFT(c, c_complex);

        // C' = C * conj(H)
        deconvolution.complexMultiplicationWithConjugate(c_complex, H, c_complex);

        // c' = IFFT(C')
        deconvolution.backwardFFT(c_complex, c);

        // d) Apply adaptive damping: c = c * a
        deconvolution.scalarMultiplication(c, static_cast<real_t>(a), c);

        // fn+1' = fn * c'
        deconvolution.multiplication(f, c, f);

        backend->sync();
        progressFunction(iterations);
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
    return 1; // Allocates 1 additional array of input size
}