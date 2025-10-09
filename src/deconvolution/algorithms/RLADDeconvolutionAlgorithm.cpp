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
    assert(backend->isOnDevice(f.data) && "PSF is not on device");
    ComplexData c = backend->allocateMemoryOnDevice(g.size);
    backend->memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {
        // Calculate damping factor
        double a;
        if (dampingDecrease == 0) {
            a = alpha * exp(-beta * n);
        } else {  // Linear decay
            a = alpha - beta * n;
        }

        std::cout << "\r[STATUS] Iteration: " << n + 1 << "/" << iterations << " ";

        // a) First transformation: Fn = FFT(fn)
        backend->forwardFFT(f, f);

        // Fn' = Fn * H
        backend->complexMultiplication(f, H, c);

        // fn' = IFFT(Fn') + NORMALIZE
        backend->backwardFFT(c, c);

        // b) Calculation of the Correction Factor: c = g / fn'
        backend->complexDivision(g, c, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        backend->forwardFFT(c, c);

        // C' = C * conj(H)
        backend->complexMultiplicationWithConjugate(c, H, c);

        // c' = IFFT(C') + NORMALIZE
        backend->backwardFFT(c, c);

        // d) Update the estimated image:
        backend->backwardFFT(f, f);

        // Apply adaptive damping: c = c * a
        backend->scalarMultiplication(c, a, c);

        // fn+1' = fn * c
        backend->complexMultiplication(f, c, f);
        
        std::flush(std::cout);
    }
    
    backend->freeMemoryOnDevice(c);
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
