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
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");
    ComplexData H2 = backend->getMemoryManager().allocateMemoryOnDevice(H.size);
    ComplexData L = backend->getMemoryManager().allocateMemoryOnDevice(H.size);
    ComplexData L2 = backend->getMemoryManager().allocateMemoryOnDevice(H.size);
    ComplexData FA = backend->getMemoryManager().allocateMemoryOnDevice(H.size);
    ComplexData FP = backend->getMemoryManager().allocateMemoryOnDevice(H.size);

    backend->getMemoryManager().memCopy(g, f);
    // Forward FFT on image
    backend->getDeconvManager().forwardFFT(f, f);

    // H*H
    backend->getDeconvManager().complexMultiplication(H, H, H2);
    
    // Laplacian L
    backend->getDeconvManager().calculateLaplacianOfPSF(H, L);
    backend->getDeconvManager().complexMultiplication(L, L, L2);
    backend->getDeconvManager().scalarMultiplication(L2, lambda, L2);

    backend->getDeconvManager().complexAddition(H2, L2, FA);
    backend->getDeconvManager().complexDivisionStabilized(H, FA, FP, complexDivisionEpsilon);
    backend->getDeconvManager().complexMultiplication(f, FP, f);

    // Inverse FFT
    backend->getDeconvManager().backwardFFT(f, f);
    backend->getDeconvManager().octantFourierShift(f);   

}

std::unique_ptr<DeconvolutionAlgorithm> RegularizedInverseFilterDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->lambda = this->lambda;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RegularizedInverseFilterDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 8; // Allocates 5 additional arrays of input size (H2, L, L2, FA, FP) + 3 input copies
}
