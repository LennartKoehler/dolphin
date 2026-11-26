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
    configure(asConfig<PoissonMLEConfig>(config, DeconvolutionAlgorithms::kPoissonMLEAdaptiveStep));
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    // Base class: the in-place real FFT plans, which the model/hd FFTs below reuse.
    PoissonMLEDeconvolutionAlgorithm::init(dataSize);
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::preAllocate(const CuboidShape& dataSize) {
    // Base class: the streamed-step staging slab (unused here, but tiny).
    PoissonMLEDeconvolutionAlgorithm::preAllocate(dataSize);

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    // Classic-path work buffer: the line search needs m and the direction d
    // alive simultaneously, so this variant never uses the streamed step and
    // owns the full work volume plus its complex view.
    scratch = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    scratchC = memory.reinterpret(scratch);

    model = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize); // m, kept alive for J(alpha)
    hd = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);    // H*d for the line search
    hdC = memory.reinterpret(hd);
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::resetDataStepState() {
    pwr = 1.0f;
}

void PoissonMLEAdaptiveStepDeconvolutionAlgorithm::applyDataStep(
    const IComputeBackend& compute, const ComplexData& H, const RealData& g,
    real_t bg, RealData& f) {
    // m = Hf + bg in this class's own buffer: the line search still needs it after
    // the direction computation, which reuses the base's `scratch` for d.
    computeForwardModel(compute, H, f, bg, model);
    computeSearchDirection(compute, H, g, f, model, scratch); // model stays intact

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    // hd = H*d  (spatial result in `hd`)
    memory.memCopy(scratch, hd);                 // hd <- d
    compute.forwardFFT(hd, hdC);                 // in-place r2c -> hdC = FFT(d)
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

    // ---- Additive update f <- f + alpha*pwr*d, clipped at zero. ----
    compute.realLinearCombination(f, 1.0f, scratch, alpha * pwr, f);
    compute.realClampMin(f, 0.0f, f);
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEAdaptiveStepDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEAdaptiveStepDeconvolutionAlgorithm>();
    copy->iterations = iterations;
    copy->signal_to_noise = signal_to_noise;
    copy->lambdaOverride = lambdaOverride;
    copy->backgroundOverride = backgroundOverride;
    copy->usePrefilter = usePrefilter;
    copy->prefilterScale = prefilterScale;
    copy->useSieve = useSieve;
    copy->usePrior = usePrior;
    copy->sievePasses = sievePasses;
    copy->sieveScale = sieveScale;
    copy->psfSigmasValid = false;
    copy->initialized = false;
    return copy;
}

size_t PoissonMLEAdaptiveStepDeconvolutionAlgorithm::getWorkingSetBytes(const CuboidShape& dataSize) const {
    // Base: scratch. This class adds model + hd.
    return PoissonMLEDeconvolutionAlgorithm::getWorkingSetBytes(dataSize)
         + 2 * dataSize.getVolume() * sizeof(real_t);
}
