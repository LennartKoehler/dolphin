#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include <iostream>
#include <omp.h>
#include <fftw3.h>
#include <cassert>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;


}

void RLDeconvolutionAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Richardson-Lucy algorithm" << std::endl;
        return;
    }

    // Allocate memory for intermediate arrays
    assert(backend->isOnDevice(f.data) && "PSF is not on device");
    ComplexData c = backend->allocateMemoryOnDevice(g.size);
    backend->memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {

        // a) First transformation: Fn = FFT(fn)
        backend->forwardFFT(f, f);

        // Fn' = Fn * H
        backend->complexMultiplication(f, H, c);

        // fn' = IFFT(Fn') + NORMALIZE
        backend->backwardFFT(c, c);
        // backend->scalarMultiplication(c, 1.0 / g.size.volume, c); // Add normalization


        // b) Calculation of the Correction Factor: c = g / fn'
        backend->complexDivision(g, c, c, complexDivisionEpsilon);

        // // c) Second transformation: C = FFT(c)
        backend->forwardFFT(c, c);

        // // C' = C * conj(H)
        backend->complexMultiplicationWithConjugate(c, H, c);

        // // c' = IFFT(C') + NORMALIZE
        backend->backwardFFT(c, c);
        // backend->scalarMultiplication(c, 1.0 / g.size.volume, c); // Add normalization


        backend->backwardFFT(f, f);
        // backend->scalarMultiplication(f, 1.0 / g.size.volume, f); // Add normalization

        backend->complexMultiplication(f, c, f);
 
    }
    backend->freeMemoryOnDevice(c);
}




std::unique_ptr<DeconvolutionAlgorithm> RLDeconvolutionAlgorithm::clone() const {
    auto copy = std::make_unique<RLDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    // Don't copy backend - each thread needs its own
    return copy;
    
}