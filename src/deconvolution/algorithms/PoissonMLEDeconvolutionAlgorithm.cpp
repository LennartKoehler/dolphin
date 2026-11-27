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

#include "dolphin/deconvolution/algorithms/PoissonMLEDeconvolutionAlgorithm.h"
#include <cassert>
#include <algorithm>

void PoissonMLEDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Accept both PoissonMLEConfig and PoissonMLEUnitStepConfig (the latter is
    // derived from the former).
    core_.configure(asConfig<PoissonMLEConfig>(config, DeconvolutionAlgorithms::kPoissonMLE));
}

void PoissonMLEDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE algorithm initialization");
    core_.initPlans(backend->mutableComputeManager(), dataSize);
    initialized = true;
}

void PoissonMLEDeconvolutionAlgorithm::preAllocate(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE algorithm allocation");
    assert(initialized && "PoissonMLE algorithm not initialized. Call init() first.");
    core_.allocateWork(backend->getMemoryManager(), dataSize);
}

bool PoissonMLEDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void PoissonMLEDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f,
                                                  const DeconvolutionRunContext& runContext) {
    assert(backend && "No backend available for PoissonMLE algorithm");
    assert(initialized && "PoissonMLE algorithm not initialized. Call init() first.");
    if (core_.iterations <= 0) return;

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& compute = backend->getComputeManager();

    assert(memory.isOnDevice(f) && "f is not on device");
    assert(memory.isOnDevice(g) && "g is not on device");

    // ---- Initial estimate: the data, optionally prefiltered. ----
    memory.memCopy(g, f);

    // ---- Physical parameters: cached PSF sigma widths (run-constant). ----
    core_.ensurePsfSigmas(compute, memory, H, core_.scratchC, core_.scratch);

    // ---- Prefilter: a single mild anisotropic Gaussian low-pass of the raw data. ----
    core_.applyPrefilter(compute, f, core_.psfSigX, core_.psfSigY, core_.psfSigZ);

    const real_t bg = core_.resolveBackground(runContext);
    const real_t lambda = core_.currentLambda();

    // f is the object ABOVE the background (m = Hf + bg), so the physical
    // constraint is f >= 0 and the background of f converges to 0 (where the
    // model m -> bg matches the observed background g -> bg). Clamping at bg
    // instead would pin the background of f at bg and make the model predict
    // 2*bg there — the background can never be fit.
    compute.realClampMin(f, 0.0f, f);

    const int maxIter = core_.iterations;
    const real_t kLast = std::max(1.0f, 0.8f * static_cast<real_t>(maxIter));

    for (int k = 0; k < maxIter; ++k) {
        // ---- DATA STEP: Poisson likelihood gradient, RL unit step. ----
        dataStep(compute, H, g, bg, f);

        // ---- PRIOR + SIEVE: regularization, then the physical clamp. ----
        core_.applyRegularization(compute, f, k, kLast, lambda);

        backend->sync();
        if (progressFunction) progressFunction(maxIter);
    }
}

void PoissonMLEDeconvolutionAlgorithm::dataStep(const IComputeBackend& compute,
                                                const ComplexData& H, const RealData& g,
                                                real_t bg, RealData& f) {
    // m = Hf + bg, then the ratio R = H^T(g/m), both in the shared work buffer:
    // the unit step never needs m past the ratio, so m is allowed to be
    // clobbered by the ratio computation.
    core_.computeForwardModel(compute, H, f, bg, core_.scratch);
    core_.computeRatio(compute, H, g, core_.scratch, core_.scratch);

    // Plain RL unit step: f <- max(f * R, 0) in one fused pass. This is the
    // multiplicative RL update written directly (f + d = f*R for
    // d = -f*(1-R)), which is also numerically cleaner than the
    // 1-(1-R) round trip and provably does not increase the Poisson
    // (I-divergence) objective.
    compute.poissonUpdate(core_.scratch, f, 0.0f);
}

std::unique_ptr<const MemoryComponent> PoissonMLEDeconvolutionAlgorithm::getMemoryComponent(
    const IBackendManager& backendManager, size_t workerThreads) const {
    return std::make_unique<PoissonMLEAlgorithmMemoryComponent>(backendManager, workerThreads);
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEDeconvolutionAlgorithm>();
    copy->core_ = core_;
    copy->core_.resetForClone();
    copy->initialized = false;
    return copy;
}
