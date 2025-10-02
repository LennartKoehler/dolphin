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
    assert(backend->isOnDevice(H.data) && "PSF is not on device");
    assert(backend->isOnDevice(g.data) && "Input image is not on device");
    assert(backend->isOnDevice(f.data) && "Output buffer is not on device");

    // Allocate memory for intermediate arrays
    ComplexData c = backend->allocateMemoryOnDevice(g.size);
    ComplexData gx = backend->allocateMemoryOnDevice(g.size);
    ComplexData gy = backend->allocateMemoryOnDevice(g.size);
    ComplexData gz = backend->allocateMemoryOnDevice(g.size);
    ComplexData tv = backend->allocateMemoryOnDevice(g.size);
    
    try {
        // Initialize result with input data
        backend->memCopy(g, f);

        // Calculate gradients and the Total Variation (one-time computation)
        backend->gradientX(g, gx);
        backend->gradientY(g, gy);
        backend->gradientZ(g, gz);
        backend->normalizeTV(gx, gy, gz, complexDivisionEpsilon);
        backend->gradientX(gx, gx);
        backend->gradientY(gy, gy);  
        backend->gradientZ(gz, gz);
        backend->computeTV(lambda, gx, gy, gz, tv);

        for (int n = 0; n < iterations; ++n) {
            // a) First transformation: Fn = FFT(fn)
            backend->forwardFFT(f, c);

            // Fn' = Fn * H
            backend->complexMultiplication(c, H, c);

            // fn' = IFFT(Fn')
            backend->backwardFFT(c, c);

            // b) Calculation of the Correction Factor: c = g / fn'
            backend->complexDivision(g, c, c, complexDivisionEpsilon);

            // c) Second transformation: C = FFT(c)
            backend->forwardFFT(c, c);

            // C' = C * conj(H)
            backend->complexMultiplicationWithConjugate(c, H, c);

            // c' = IFFT(C')
            backend->backwardFFT(c, c);

            // d) Update the estimated image: fn+1' = fn * c'
            backend->complexMultiplication(f, c, f);

            // fn+1 = fn+1' * tv (apply TV regularization)
            backend->complexMultiplication(f, tv, f);
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in Richardson-Lucy TV algorithm: " << e.what() << std::endl;
    }

    // Clean up allocated memory
    backend->freeMemoryOnDevice(c);
    backend->freeMemoryOnDevice(gx);
    backend->freeMemoryOnDevice(gy);
    backend->freeMemoryOnDevice(gz);
    backend->freeMemoryOnDevice(tv);
}

std::unique_ptr<DeconvolutionAlgorithm> RLTVDeconvolutionAlgorithm::clone() const {
    auto copy = std::make_unique<RLTVDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->lambda = this->lambda;
    copy->complexDivisionEpsilon = this->complexDivisionEpsilon;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLTVDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 5; // Allocates 5 additional arrays of input size (c, gx, gy, gz, tv)
}
