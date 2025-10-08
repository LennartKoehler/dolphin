#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include <iostream>
#include <cassert>

void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Set epsilon for stabilized division
    epsilon = config.epsilon;  // Assuming epsilon is in the config
}

void InverseFilterDeconvolutionAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Inverse Filter algorithm" << std::endl;
        return;
    }

    // Verify inputs are on device
    assert(backend->isOnDevice(H.data) && "PSF is not on device");
    assert(backend->isOnDevice(g.data) && "Input image is not on device");
    assert(backend->isOnDevice(f.data) && "Output buffer is not on device");

    // Allocate temporary memory for computation
    ComplexData temp_g = backend->allocateMemoryOnDevice(g.size);

    try {
        // Copy input data to working array
        backend->memCopy(g, temp_g);

        // Forward FFT on image
        backend->forwardFFT(temp_g, temp_g);

        // Division in frequency domain: F = G / H (with stabilization)
        backend->complexDivision(temp_g, H, f, epsilon);

        // Inverse FFT to get result
        backend->backwardFFT(f, f);

        // Optional: Apply normalization if needed
        // backend->scalarMultiplication(f, 1.0 / g.size.volume, f);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in inverse filter algorithm: " << e.what() << std::endl;
    }

    // Clean up allocated memory
    backend->freeMemoryOnDevice(temp_g);
}

std::unique_ptr<DeconvolutionAlgorithm> InverseFilterDeconvolutionAlgorithm::clone() const {
    auto copy = std::make_unique<InverseFilterDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->epsilon = this->epsilon;
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t InverseFilterDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 4; // Allocates 1 additional array of input size + 3 input copies
}
