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

#include "dolphin/deconvolution/algorithms/TestAlgorithm.h"
#include <cassert>
#include <spdlog/spdlog.h>

// void TestAlgorithm::configure(const DeconvolutionConfig& config) {
//     // Call base class configure to set up common parameters
//     // No specific parameters for test algorithm
// }
//
// void TestAlgorithm::init(const CuboidShape& dataSize) {
//     assert(backend && "No backend available for Test algorithm initialization");\
//
//     initialized = true;
// }
//
// bool TestAlgorithm::isInitialized() const {
//     return initialized;
// }
//

void TestAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm initialization");\

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    // Allocate memory for intermediate arrays (iteration-persistent only)
    c = std::move(memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize));
    c_complex = c.reinterpret(); // same data pointer, just reinterpreted

    // Allocate temporary complex buffer for FFT operations
    f_complex = memory.allocateMemoryOnDeviceComplex(dataSize);
    tv = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    gx = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    gy = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    gz = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    initialized = true;
}

void TestAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f) {
    assert(backend && "No backend available for Convolution algorithm");\

    assert(initialized && "Convolution algorithm not initialized. Call init() first.");\
    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& deconv = backend->getComputeManager();

    // ComplexData f_complex = memory.allocateMemoryOnDevice(f.getSize());
    // RealData c_real = memory.allocateMemoryOnDeviceReal(f.getSize());
    // deconv.forwardFFT(g, f_complex);
    // deconv.backwardFFT(H, f);
    deconv.gradient(g, gx, gy, gz);

    const real_t tvBeta = static_cast<real_t>(lambda) * static_cast<real_t>(0.1);
    deconv.normalizeTV(gx, gy, gz, tvBeta);

    deconv.divergence(gx, gy, gz, tv);
    // deconv.computeTV(lambda, tv, tv); // in-place: tv is both div input and tv output
    memory.memCopy(tv, f);
}
//
std::unique_ptr<DeconvolutionAlgorithm> TestAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<TestAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->lambda = this->lambda;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}
//
// size_t TestAlgorithm::getMemoryMultiplier() const {
//     return 0; // No additional memory allocation needed
// }
