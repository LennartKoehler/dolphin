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

#include "dolphin/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>
#include <spdlog/spdlog.h>

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Configure algorithm-specific parameters
    iterations = config.iterations;
    lambda = config.lambda;
    

}

void RLTVDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    if (!backend) {
        spdlog::error("No backend available for Richardson-Lucy TV algorithm initialization");
        return;
    }
    
    // Allocate memory for intermediate arrays
    c = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    gx = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    gy = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    gz = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    tv = std::move(backend->getMemoryManager().allocateMemoryOnDevice(dataSize));
    
    initialized = true;
}

bool RLTVDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLTVDeconvolutionAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    if (!backend) {
        spdlog::error("No backend available for Richardson-Lucy TV algorithm");
        return;
    }
    
    if (!initialized) {
        spdlog::error("Richardson-Lucy TV algorithm not initialized. Call init() first.");
        return;
    }

    // Verify inputs are on device
    assert(backend->getMemoryManager().isOnDevice(H.data) && "PSF is not on device");
    assert(backend->getMemoryManager().isOnDevice(g.data) && "Input image is not on device");
    assert(backend->getMemoryManager().isOnDevice(f.data) && "Output buffer is not on device");

    // Use pre-allocated memory for intermediate arrays
    // Initialize result with input data
    backend->getMemoryManager().memCopy(g, f);

    // Calculate gradients and the Total Variation (one-time computation)

    backend->getDeconvManager().gradientX(g, gx);
    backend->getDeconvManager().gradientY(g, gy);
    backend->getDeconvManager().gradientZ(g, gz);
    backend->getDeconvManager().normalizeTV(gx, gy, gz, complexDivisionEpsilon);
    backend->getDeconvManager().gradientX(gx, gx);
    backend->getDeconvManager().gradientY(gy, gy);
    backend->getDeconvManager().gradientZ(gz, gz);
    backend->getDeconvManager().computeTV(lambda, gx, gy, gz, tv);

    for (int n = 0; n < iterations; ++n) {
        // a) First transformation: Fn = FFT(fn)
        backend->getDeconvManager().forwardFFT(f, c);

        // Fn' = Fn * H
        backend->getDeconvManager().complexMultiplication(c, H, c);

        // fn' = IFFT(Fn')
        backend->getDeconvManager().backwardFFT(c, c);

        // b) Calculation of the Correction Factor: c = g / fn'
        backend->getDeconvManager().complexDivision(g, c, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        backend->getDeconvManager().forwardFFT(c, c);

        // C' = C * conj(H)
        backend->getDeconvManager().complexMultiplicationWithConjugate(c, H, c);

        // c' = IFFT(C')
        backend->getDeconvManager().backwardFFT(c, c);

        // d) Update the estimated image: fn+1' = fn * c'
        backend->getDeconvManager().complexMultiplication(f, c, f);

        // fn+1 = fn+1' * tv (apply TV regularization)
        backend->getDeconvManager().complexMultiplication(f, tv, f);
    }


}

std::unique_ptr<DeconvolutionAlgorithm> RLTVDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLTVDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->lambda = this->lambda;
    copy->complexDivisionEpsilon = this->complexDivisionEpsilon;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLTVDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 5; // Allocates 5 additional arrays of input size (c, gx, gy, gz, tv) 
}
