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
#include <cassert>
#include <spdlog/spdlog.h>

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
    lambda = config.lambda;
}

void RLTVDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm initialization");\

    // Allocate memory for intermediate arrays
    c = std::move(backend->getMemoryManager().allocateMemoryOnDeviceReal(dataSize));
    // c_complex = std::move(backend->getMemoryManager().allocateMemoryOnDeviceComplex(dataSize));
    c_complex = c.reinterpret(); // same data pointer, just reinterpreted

    // Allocate temporary complex buffer for FFT operations
    f_complex = backend->getMemoryManager().allocateMemoryOnDeviceComplex(dataSize);
    tv = backend->getMemoryManager().allocateMemoryOnDeviceReal(dataSize);

    initialized = true;
}

bool RLTVDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLTVDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f) {
    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IDeconvolutionBackend& deconvolution = backend->getDeconvManager();

    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm");\
    assert(initialized && "Richardson-Lucy with TV regularization algorithm not initialized. Call init() first.");\
    assert(memory.isOnDevice(H.getData()) && "PSF is not on device");
    assert(memory.isOnDevice(g.getData()) && "Input image is not on device");
    assert(memory.isOnDevice(f.getData()) && "Output buffer is not on device");

    // Initialize result with input data
    memory.memCopy(g, f);

    {
        // Pre-compute TV regularization term (all in spatial domain with real-valued data)
        RealData gx = memory.allocateMemoryOnDeviceReal(g.getSize());
        RealData gy = memory.allocateMemoryOnDeviceReal(g.getSize());
        RealData gz = memory.allocateMemoryOnDeviceReal(g.getSize());
        deconvolution.gradientX(g, gx);
        deconvolution.gradientY(g, gy);
        deconvolution.gradientZ(g, gz);
        deconvolution.normalizeTV(gx, gy, gz, complexDivisionEpsilon);
        // Second pass gradients on normalized gradients
        RealData gx2 = memory.allocateMemoryOnDeviceReal(g.getSize());
        RealData gy2 = memory.allocateMemoryOnDeviceReal(g.getSize());
        RealData gz2 = memory.allocateMemoryOnDeviceReal(g.getSize());
        // if done in parallel like on gpu then not safe to do inplace
        deconvolution.gradientX(gx, gx2);
        deconvolution.gradientY(gy, gy2);
        deconvolution.gradientZ(gz, gz2);
        deconvolution.computeTV(lambda, gx2, gy2, gz2, tv);
        // RAII deallocate
    }


    for (int n = 0; n < iterations; ++n) {
        progressFunction(iterations);

        // a) First transformation: Fn = FFT(fn)
        deconvolution.forwardFFT(f, f_complex);

        // Fn' = Fn * H
        deconvolution.complexMultiplication(f_complex, H, c_complex);

        // fn' = IFFT(Fn')
        deconvolution.backwardFFT(c_complex, c);

        // b) Calculation of the Correction Factor: c = g / fn'
        deconvolution.division(g, c, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        deconvolution.forwardFFT(c, c_complex);

        // C' = C * conj(H)
        deconvolution.complexMultiplicationWithConjugate(c_complex, H, c_complex);

        // c' = IFFT(C')
        deconvolution.backwardFFT(c_complex, c);

        // d) Update the estimated image: fn+1' = fn * c'
        deconvolution.multiplication(f, c, f);

        // fn+1 = fn+1' * tv (apply TV regularization)
        deconvolution.multiplication(f, tv, f);

        // backend->sync();
    }
}

std::unique_ptr<DeconvolutionAlgorithm> RLTVDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLTVDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->lambda = this->lambda;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t RLTVDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 4; // Allocates 4 additional arrays of input size
}
