#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include <iostream>
#include <omp.h>
#include <fftw3.h>
#include <cassert>

void RLADDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
    dampingDecrease = 0; // Fixed to exponential decay as in original
    alpha = 0.9;         // Fixed as in original
    beta = 0.01;         // Fixed as in original
}

void RLADDeconvolutionAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for RLAD algorithm" << std::endl;
        return;
    }

    // Allocate memory for intermediate arrays
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");
    ComplexData c = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
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
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLADDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 4; // Allocates 1 additional array of input size + 3 input copies
}
