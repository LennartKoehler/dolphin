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

#include "dolphin/deconvolution/algorithms/PoissonMLEAdaptiveStepDeconvolutionAlgorithm.h"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

const real_t kAlphaLow = 0.005f;
const real_t kAlphaHigh = 25.0f;

// Safeguarded Newton (bisection fallback) for the 1-D minimizer of the Poisson
// line search J(alpha) = sum g ln(g/(m+alpha*hd)) - g + (m+alpha*hd), which is
// strictly convex in alpha. The result is further guarded against Newton overshoot
// by preferring, over {newton alpha, the RL unit step (alpha=1)}, whichever gives
// the smaller J: the RL step is the provably-safe descent, while the exact Newton
// optimum is faster when it beats it.
real_t newtonLineSearch(const IComputeBackend& compute,
                        const RealData& g, const RealData& m, const RealData& hd) {
    real_t a = kAlphaLow;
    real_t J = 0, dJ = 0, d2J = 0;
    compute.poissonLineSearchEval(g, m, hd, a, &J, &dJ, &d2J);
    a = (d2J > 1e-9) ? std::clamp(a - dJ / d2J, kAlphaLow, kAlphaHigh) : a;

    for (int it = 0; it < 30; ++it) {
        compute.poissonLineSearchEval(g, m, hd, a, &J, &dJ, &d2J);
        real_t aNext = (d2J > 1e-9) ? (a - dJ / d2J) : 0.5 * (kAlphaLow + kAlphaHigh);
        if (!(aNext > kAlphaLow && aNext < kAlphaHigh)) aNext = 0.5 * (kAlphaLow + kAlphaHigh);
        if (std::fabs(aNext - a) < 1e-5 * (1.0 + std::fabs(a))) {
            a = aNext;
            break;
        }
        a = aNext;
    }
    a = std::clamp(a, kAlphaLow, kAlphaHigh);

    // Guard: never do worse than the RL unit step.
    real_t Jnew = 0, Jrl = 0, t = 0;
    compute.poissonLineSearchEval(g, m, hd, a, &Jnew, &t, &t);
    compute.poissonLineSearchEval(g, m, hd, 1.0f, &Jrl, &t, &t);
    return (Jnew <= Jrl) ? a : 1.0f;
}

} // namespace

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Accept both PoissonMLEConfig and PoissonMLEUnitStepConfig (the latter is
    // derived from the former).
    core_.configure(asConfig<PoissonMLEConfig>(config, DeconvolutionAlgorithms::kPoissonMLEAdaptiveStep));
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE (adaptive) algorithm initialization");
    // The in-place real FFT plans, which the model/hd FFTs below reuse.
    core_.initPlans(backend->mutableComputeManager(), dataSize);
    initialized = true;
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::preAllocate(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE (adaptive) algorithm allocation");
    assert(initialized && "PoissonMLE (adaptive) algorithm not initialized. Call init() first.");
    // The shared work buffer (carries the direction d; also the sieve OTF buffer).
    core_.allocateWork(backend->getMemoryManager(), dataSize);

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    model = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize); // m, kept alive for J(alpha)
    hd = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);    // H*d for the line search
    hdC = memory.reinterpret(hd);
}

bool PoissonMLEAdaptiveStepDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f,
                                                              const DeconvolutionRunContext& runContext) {
    assert(backend && "No backend available for PoissonMLE (adaptive) algorithm");
    assert(initialized && "PoissonMLE (adaptive) algorithm not initialized. Call init() first.");
    if (core_.iterations <= 0) return;

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& compute = backend->getComputeManager();
    const CuboidShape shape = f.getSize();

    assert(memory.isOnDevice(f) && "f is not on device");
    assert(memory.isOnDevice(g) && "g is not on device");

    // ---- Initial estimate: the data, optionally prefiltered. ----
    memory.memCopy(g, f);

    // ---- Physical parameters: cached PSF sigma widths (run-constant). ----
    core_.ensurePsfSigmas(compute, memory, H, shape, core_.scratchC, core_.scratch);

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
    pwr = 1.0f; // reset per-deconvolution step-damping state

    for (int k = 0; k < maxIter; ++k) {
        // ---- DATA STEP: exact Newton line search on the Poisson objective. ----
        dataStep(compute, H, g, bg, f);

        // ---- PRIOR + SIEVE: regularization, then the physical clamp. ----
        core_.applyRegularization(compute, f, k, kLast, lambda);

        backend->sync();
        if (progressFunction) progressFunction(maxIter);
    }
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::dataStep(const IComputeBackend& compute,
                                                            const ComplexData& H, const RealData& g,
                                                            real_t bg, RealData& f) {
    // m = Hf + bg in its OWN buffer: the line search still needs it after the
    // direction computation, which reuses core_.scratch for d.
    core_.computeForwardModel(compute, H, f, bg, model);
    core_.computeSearchDirection(compute, H, g, f, model, core_.scratch);

    // hd = H*d  (spatial result in `hd`). The r2c runs OUT-OF-PLACE from the
    // padded direction buffer directly into hd's complex view — no full-volume
    // copy of d into hd per iteration.
    compute.forwardFFT(core_.scratch, hdC);      // out-of-place r2c -> hdC = FFT(d)
    compute.complexMultiplication(hdC, H, hdC);
    compute.backwardFFT(hdC, hd);                // in-place c2r -> hd = H*d

    // current objective and optimal step
    real_t Jcur = 0, dJ = 0, d2J = 0;
    compute.poissonLineSearchEval(g, model, hd, 0.0f, &Jcur, &dJ, &d2J);
    const real_t alpha = newtonLineSearch(compute, g, model, hd);
    real_t Jnew = 0;
    compute.poissonLineSearchEval(g, model, hd, alpha, &Jnew, &dJ, &d2J);

    // ---- GUARD: if the likelihood worsened, damp the update power. ----
    if (Jnew > Jcur && pwr > 0.25f) {
        pwr *= 0.5f;
    }

    // ---- Additive update f <- max(f + alpha*pwr*d, 0) in one fused pass. ----
    compute.realLinearCombinationClampMin(f, 1.0f, core_.scratch, alpha * pwr, 0.0f, f);
}

std::unique_ptr<const MemoryComponent> PoissonMLEAdaptiveStepDeconvolutionAlgorithm::getMemoryComponent(
    const IBackendManager& backendManager, size_t workerThreads) const {
    return std::make_unique<PoissonMLEAdaptiveStepAlgorithmMemoryComponent>(backendManager, workerThreads);
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEAdaptiveStepDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEAdaptiveStepDeconvolutionAlgorithm>();
    copy->core_ = core_;
    copy->core_.resetForClone();
    copy->initialized = false;
    return copy;
}