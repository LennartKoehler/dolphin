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
#include <cassert>
#include <spdlog/spdlog.h>

void RegularizedInverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    lambda = config.lambda;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Regularized Inverse Filter algorithm initialization");\

    // Allocate memory for intermediate arrays (all in frequency domain)
    H2 = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    L = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    L2 = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    FA = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    FP = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    f_complex = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));

    initialized = true;
}

bool RegularizedInverseFilterDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f) {
    assert(backend && "No backend available for Regularized Inverse Filter algorithm");\

    assert(initialized && "Regularized Inverse Filter algorithm not initialized. Call init() first.");\

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IDeconvolutionBackend& deconvolution = backend->getDeconvManager();

    complex_t lambdacomplex = {static_cast<real_t>(lambda), 0};

    // Verify inputs are on device
    assert(memory.isOnDevice(H.data) && "PSF is not on device");
    assert(memory.isOnDevice(g.data) && "Input image is not on device");
    assert(memory.isOnDevice(f.data) && "Output buffer is not on device");

    // Copy input to output buffer
    memory.memCopy(g, f);

    // Forward FFT on image: RealData -> ComplexData
    deconvolution.forwardFFT(f, f_complex);

    // H*H
    deconvolution.complexMultiplication(H, H, H2);

    // Laplacian L
    deconvolution.calculateLaplacianOfPSF(H, L);
    deconvolution.complexMultiplication(L, L, L2);
    deconvolution.scalarMultiplication(L2, lambdacomplex, L2);

    deconvolution.complexAddition(H2, L2, FA);
    deconvolution.complexDivisionStabilized(H, FA, FP, complexDivisionEpsilon);
    deconvolution.complexMultiplication(f_complex, FP, f_complex);

    // Inverse FFT: ComplexData -> RealData
    deconvolution.backwardFFT(f_complex, f);
    deconvolution.octantFourierShift(f);
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
    return 5; // Allocates 5 additional arrays of input size (H2, L, L2, FA, FP, f_complex - but FP can be reused)
}
