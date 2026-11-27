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
#include "dolphinbackend/IComputeBackend.h"
#include <cassert>
#include <spdlog/spdlog.h>


// Shrinkage base: lambda = kLambdaBase / snr (0 = derive from SNR). Higher SNR
// -> smaller lambda -> weaker prior (closer to the pure MLE).
const real_t kLambdaBase = 1.0f;

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    configure(asConfig<RichardsonLucyTVConfig>(config, DeconvolutionAlgorithms::kRichardsonLucyTotalVariation));
}

void RLTVDeconvolutionAlgorithm::configure(const RichardsonLucyTVConfig& config) {
    iterations = config.iterations;
    signal_to_noise = config.signalToNoise;
}


void RLTVDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm initialization");

    initialized = true;
}

void RLTVDeconvolutionAlgorithm::preAllocate(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm allocation");
    assert(initialized && "Richardson-Lucy with TV regularization algorithm not initialized. Call init() first.");

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    // Allocate memory for intermediate arrays (iteration-persistent only)
    c = std::move(memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize));
    c_complex = c.reinterpret(); // same data pointer, just reinterpreted

    tv = memory.allocateMemoryOnDeviceReal(dataSize);
    gx = memory.allocateMemoryOnDeviceReal(dataSize);
    gy = memory.allocateMemoryOnDeviceReal(dataSize);
    gz = memory.allocateMemoryOnDeviceReal(dataSize);
}

bool RLTVDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void RLTVDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f,
    const DeconvolutionRunContext& /*runContext*/) {
    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& deconvolution = backend->getComputeManager();

    assert(backend && "No backend available for Richardson-Lucy with TV regularization algorithm");\
    assert(initialized && "Richardson-Lucy with TV regularization algorithm not initialized. Call init() first.");\
    assert(memory.isOnDevice(H) && "PSF is not on device");
    assert(memory.isOnDevice(g) && "Input image is not on device");
    assert(memory.isOnDevice(f) && "Output buffer is not on device");

    // Initialize result with input data
    memory.memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {
        if (progressFunction) progressFunction(iterations);

        // a) First transformation: Fn = FFT(fn)
        deconvolution.forwardFFT(f, c_complex);

        // Fn' = Fn * H (in-place: c_complex * H -> c_complex)
        deconvolution.complexMultiplication(c_complex, H, c_complex);

        // fn' = IFFT(Fn')
        deconvolution.backwardFFT(c_complex, c);

        // b) Calculation of the Correction Factor: c = g / fn'
        deconvolution.division(g, c, c, static_cast<real_t>(min_denominator));

        // c) Second transformation: C = FFT(c)
        deconvolution.forwardFFT(c, c_complex);

        // C' = C * conj(H)
        deconvolution.complexMultiplicationWithConjugate(c_complex, H, c_complex);

        // c' = IFFT(C')
        deconvolution.backwardFFT(c_complex, c);

        // d) Update the estimated image: fn+1' = fn * c'
        deconvolution.multiplication(f, c, f);

        // fn+1 = fn+1' / (1 - lambda * div) — TV regularization (Dey et al.)
        // div is the divergence of the normalized gradient of f. At edges/noise
        // spikes, div < 0, so (1 - lambda*div) > 1, damping those regions.
        // tv = 1/(1-lambda*div), so f * tv = f/(1-lambda*div).
        computeTV(f);
        deconvolution.multiplication(f, tv, f);

        // backend->sync();
    }
}

std::unique_ptr<DeconvolutionAlgorithm> RLTVDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<RLTVDeconvolutionAlgorithm>();
    // Copy all relevant state
    copy->iterations = this->iterations;
    copy->signal_to_noise = this->signal_to_noise;
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

std::unique_ptr<const MemoryComponent> RLTVDeconvolutionAlgorithm::getMemoryComponent(
    const IBackendManager& backendManager, size_t workerThreads) const {
    return std::make_unique<RLTVAlgorithmMemoryComponent>(backendManager, workerThreads);
}

void RLTVDeconvolutionAlgorithm::computeTV(const RealData& g){
    const IComputeBackend& deconvolution = backend->getComputeManager();

    deconvolution.gradient(g, gx, gy, gz);

    const real_t tvBeta = static_cast<real_t>(1e-6);
    deconvolution.normalizeTV(gx, gy, gz, tvBeta);

    deconvolution.divergence(gx, gy, gz, tv);

    // SNR -> shrinkage strength lambda. 0 = derive from SNR (lambda = base/snr:
    // higher SNR -> weaker prior, closer to the pure MLE).
    const real_t lambda = kLambdaBase / std::max(1e-6f, signal_to_noise);
    deconvolution.computeTV(static_cast<real_t>(lambda), tv, tv); // in-place: tv is both div input and tv output
}
