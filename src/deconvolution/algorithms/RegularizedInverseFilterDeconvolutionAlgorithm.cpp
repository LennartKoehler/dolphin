#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include <iostream>
#include <omp.h>
#include <fftw3.h>
#include <cassert>

void RegularizedInverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    lambda = config.lambda;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Regularized Inverse Filter algorithm" << std::endl;
        return;
    }

    // Allocate memory for intermediate arrays
    assert(backend->isOnDevice(f.data) && "PSF is not on device");
    ComplexData H2 = backend->allocateMemoryOnDevice(H.size);
    ComplexData L = backend->allocateMemoryOnDevice(H.size);
    ComplexData L2 = backend->allocateMemoryOnDevice(H.size);
    ComplexData FA = backend->allocateMemoryOnDevice(H.size);
    ComplexData FP = backend->allocateMemoryOnDevice(H.size);

    backend->memCopy(g, f); 
    try {
        // Forward FFT on image
        backend->forwardFFT(f, f);

        // H*H
        backend->complexMultiplication(H, H, H2);
        
        // Laplacian L
        backend->calculateLaplacianOfPSF(H, L);
        backend->complexMultiplication(L, L, L2);
        backend->scalarMultiplication(L2, lambda, L2);

        backend->complexAddition(H2, L2, FA);
        backend->complexDivisionStabilized(H, FA, FP, complexDivisionEpsilon);
        backend->complexMultiplication(f, FP, f);

        // Inverse FFT
        backend->backwardFFT(f, f);
        backend->octantFourierShift(f);
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in regularized inverse filter algorithm: " << e.what() << std::endl;
    }
    
    // Cleanup allocated arrays
    backend->freeMemoryOnDevice(H2);
    backend->freeMemoryOnDevice(L);
    backend->freeMemoryOnDevice(L2);
    backend->freeMemoryOnDevice(FA);
    backend->freeMemoryOnDevice(FP);
}

std::unique_ptr<DeconvolutionAlgorithm> RegularizedInverseFilterDeconvolutionAlgorithm::clone() const {
    auto copy = std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->lambda = this->lambda;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RegularizedInverseFilterDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 5; // Allocates 5 additional arrays of input size (H2, L, L2, FA, FP)
}
