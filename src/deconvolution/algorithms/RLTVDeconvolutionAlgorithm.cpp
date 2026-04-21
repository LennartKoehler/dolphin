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
#include "dolphinbackend/IDeconvolutionBackend.h"
#include <cassert>
#include <spdlog/spdlog.h>

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;
    lambda = config.lambda;
}


void RLTVDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm initialization");\

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    // Allocate memory for intermediate arrays (iteration-persistent only)
    c = std::move(memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize));
    c_complex = c.reinterpret(); // same data pointer, just reinterpreted

    // Allocate temporary complex buffer for FFT operations
    f_complex = memory.allocateMemoryOnDeviceComplex(dataSize);
    tv = memory.allocateMemoryOnDeviceReal(dataSize);
    gx = memory.allocateMemoryOnDeviceReal(dataSize);
    gy = memory.allocateMemoryOnDeviceReal(dataSize);
    gz = memory.allocateMemoryOnDeviceReal(dataSize);

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

        computeTV(f);
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
    return 6; // Allocates 4 additional arrays of input size
}

void RLTVDeconvolutionAlgorithm::computeTV(const RealData& g){
    const IDeconvolutionBackend& deconvolution = backend->getDeconvManager();

    deconvolution.gradientX(g, gx);
    deconvolution.gradientY(g, gy);
    deconvolution.gradientZ(g, gz);
    deconvolution.normalizeTV(gx, gy, gz, complexDivisionEpsilon);

    // dont do in place because they need neighbor acces so if its done in parallel its ub
    // use tv as scratch buffer
    deconvolution.gradientX(gx, tv);
    deconvolution.gradientY(gy, gx);
    deconvolution.gradientZ(gz, gy);

    deconvolution.computeTV(lambda, tv, gx, gy, tv); //can be done inplace
}
