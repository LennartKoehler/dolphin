#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Configure algorithm-specific parameters
    iterations = config.iterations;
    lambda = config.lambda;
    
    std::cout << "[CONFIGURATION] Richardson-Lucy Total Variation algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << iterations << std::endl;
    std::cout << "[CONFIGURATION] lambda: " << lambda << std::endl;
}

void RLTVDeconvolutionAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Richardson-Lucy TV algorithm" << std::endl;
        return;
    }

    // Verify inputs are on device
    assert(backend->getMemoryManager().isOnDevice(H.data) && "PSF is not on device");
    assert(backend->getMemoryManager().isOnDevice(g.data) && "Input image is not on device");
    assert(backend->getMemoryManager().isOnDevice(f.data) && "Output buffer is not on device");

    // Allocate memory for intermediate arrays
    ComplexData c = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    ComplexData gx = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    ComplexData gy = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    ComplexData gz = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    ComplexData tv = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    
    try {
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

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in Richardson-Lucy TV algorithm: " << e.what() << std::endl;
    }

    // Clean up allocated memory
    backend->getMemoryManager().freeMemoryOnDevice(c);
    backend->getMemoryManager().freeMemoryOnDevice(gx);
    backend->getMemoryManager().freeMemoryOnDevice(gy);
    backend->getMemoryManager().freeMemoryOnDevice(gz);
    backend->getMemoryManager().freeMemoryOnDevice(tv);
}

std::unique_ptr<DeconvolutionAlgorithm> RLTVDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLTVDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->lambda = this->lambda;
    copy->complexDivisionEpsilon = this->complexDivisionEpsilon;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLTVDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 8; // Allocates 5 additional arrays of input size (c, gx, gy, gz, tv) + 3 input copies
}
