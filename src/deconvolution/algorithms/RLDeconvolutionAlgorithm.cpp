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
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");
    ComplexData c = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    backend->getMemoryManager().memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {

        // a) First transformation: Fn = FFT(fn)
        backend->getDeconvManager().forwardFFT(f, f);

        // Fn' = Fn * H
        backend->getDeconvManager().complexMultiplication(f, H, c);

        // fn' = IFFT(Fn') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);
        // backend->getDeconvManager().scalarMultiplication(c, 1.0 / g.size.volume, c); // Add normalization


        // b) Calculation of the Correction Factor: c = g / fn'
        backend->getDeconvManager().complexDivision(g, c, c, complexDivisionEpsilon);

        // // c) Second transformation: C = FFT(c)
        backend->getDeconvManager().forwardFFT(c, c);

        // // C' = C * conj(H)
        backend->getDeconvManager().complexMultiplicationWithConjugate(c, H, c);

        // // c' = IFFT(C') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);
        // backend->getDeconvManager().scalarMultiplication(c, 1.0 / g.size.volume, c); // Add normalization


        backend->getDeconvManager().backwardFFT(f, f);
        // backend->getDeconvManager().scalarMultiplication(f, 1.0 / g.size.volume, f); // Add normalization

        backend->getDeconvManager().complexMultiplication(f, c, f);
 
    }
    // backend->getMemoryManager().freeMemoryOnDevice(c); // dont need because it is managed within complexdatas destructor
}




std::unique_ptr<DeconvolutionAlgorithm> RLDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 4; // Allocates 1 additional array of input size + 3 input copies
}