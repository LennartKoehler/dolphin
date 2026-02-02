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

#include "dolphin/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include <iostream>

#include <cassert>
#include <spdlog/spdlog.h>

void RegularizedInverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    lambda = config.lambda;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    if (!backend) {
        spdlog::error("No backend available for Regularized Inverse Filter algorithm initialization");
        return;
    }
    
    // Allocate memory for intermediate arrays
    H2 = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    L = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    L2 = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    FA = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    FP = backend->getMemoryManager().allocateMemoryOnDevice(dataSize);
    
    initialized = true;
}

bool RegularizedInverseFilterDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    if (!backend) {
        spdlog::error("No backend available for Regularized Inverse Filter algorithm");
        return;
    }
    
    if (!initialized) {
        spdlog::error("Regularized Inverse Filter algorithm not initialized. Call init() first.");
        return;
    }
    complex_t lambdacomplex = {static_cast<real_t>(lambda), 0};
    // Use pre-allocated memory for intermediate arrays
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");

    backend->getMemoryManager().memCopy(g, f);
    // Forward FFT on image
    backend->getDeconvManager().forwardFFT(f, f);

    // H*H
    backend->getDeconvManager().complexMultiplication(H, H, H2);
    
    // Laplacian L
    backend->getDeconvManager().calculateLaplacianOfPSF(H, L);
    backend->getDeconvManager().complexMultiplication(L, L, L2);
    backend->getDeconvManager().scalarMultiplication(L2, lambdacomplex, L2);

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
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RegularizedInverseFilterDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 5; // Allocates 5 additional arrays of input size (H2, L, L2, FA, FP)
}
