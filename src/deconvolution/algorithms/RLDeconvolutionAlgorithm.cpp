#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include <iostream>
#include <omp.h>
#include <fftw3.h>
#include <cassert>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;


}

// Legacy algorithm method for compatibility with existing code
void RLDeconvolutionAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Richardson-Lucy algorithm" << std::endl;
        return;
    }

    // Allocate memory for intermediate arrays
    assert(backend->isOnDevice(f.data) + "PSF is not on device");
    ComplexData c = backend->allocateMemoryOnDevice(g.size);

    for (int n = 0; n < iterations; ++n) {
        std::cerr << "\r[STATUS] Iteration: " << n << " ";


        // a) First transformation:Fn = FFT(fn)
        backend->forwardFFT(f, c);

        
        // Fn' = Fn * H
        backend->complexMultiplication(f, H, c);

        // fn' = IFFT(Fn')
        backend->backwardFFT(c, f);

        backend->octantFourierShift(f);

        // b) Calculation of the Correction Factor: c = g / fn'
        backend->complexDivision(g, f, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        backend->forwardFFT(c, f);


        // C' = C * conj(H)
        backend->complexMultiplicationWithConjugate(f, H, f);

        // c' = IFFT(C')
        backend->backwardFFT(f, c);

        backend->octantFourierShift(c);

        // d) Update the estimated image: fn+1 = fn * c
        backend->memCopy(c, f);



        std::flush(std::cout);
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