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
#include "dolphin/backend/BackendFactory.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <spdlog/spdlog.h>

namespace {

// Shrinkage base: lambda = kLambdaBase / snr (0 = derive from SNR). Higher SNR
// -> smaller lambda -> weaker prior (closer to the pure MLE).
const real_t kLambdaBase = 1.0f;

// Annealed anisotropic Gaussian smoothing in the frequency domain (the "sieve").
// Operates in-place on `f` (which has the in-place-FFT layout): r2c, multiply by the
// (half-complex) Gaussian OTF, c2r. `f` is both the source and the output.
void applySieve(const IComputeBackend& compute, RealData& f, ComplexData& sieveOTF,
                real_t sigmaX, real_t sigmaY, real_t sigmaZ) {
    if (sigmaX < 0.05f && sigmaY < 0.05f && sigmaZ < 0.05f) {
        return; // smoothing width ~ 0, sieve is the identity
    }
    ComplexView fv = f.reinterpret();       // non-owning complex view of f
    compute.fillGaussianOTF(sieveOTF, sigmaX, sigmaY, sigmaZ);
    compute.forwardFFT(f, fv);              // in-place r2c -> f (frequency domain)
    compute.complexMultiplication(fv, sieveOTF, fv);
    compute.backwardFFT(fv, f);             // in-place c2r -> f (smoothed, spatial)
}

// Estimate the PSF's lateral/axial width (voxels) from the OTF by recovering the
// centered spatial PSF (in-place c2r into the `work`/`workReal` buffer) and taking
// the standard deviation of its positive mass.
void estimatePsfSigmas(const IComputeBackend& compute,
                       const IBackendMemoryManager& mem,
                       const ComplexData& H, const CuboidShape& shape,
                       ComplexData& work, RealData& workReal,
                       real_t& sigX, real_t& sigY, real_t& sigZ) {
    // work = H (copy), then in-place c2r -> spatial PSF in workReal.
    mem.memCopy(H, work);
    compute.backwardFFT(work, workReal);

    const size_t W = shape.width, Hh = shape.height, D = shape.depth;

    // Copy the recovered (device) PSF to the host before the moment loop — the
    // host pointer must never be dereferenced while it still points at device
    // memory (CUDA). Same copy-to-host idiom as estimateBackground; the host
    // copy preserves the padded in-place layout, which access() maps back to
    // logical indices (per-row padding is added by convertIndex).
    const IBackendMemoryManager& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();
    RealData psfCopy = mem.copyDataToDevice(workReal);
    RealData psfGhost = mem.moveDataFromDevice(psfCopy, hostMem);

    double total = 0, mx = 0, my = 0, mz = 0, mx2 = 0, my2 = 0, mz2 = 0;
    for (size_t z = 0; z < D; ++z)
        for (size_t y = 0; y < Hh; ++y)
            for (size_t x = 0; x < W; ++x) {
                real_t v = psfGhost.access(z * W * Hh + y * W + x);
                if (v > 0.0f) {
                    // The OTF comes from an octant-shifted PSF, so the recovered spatial
                    // PSF is corner-peaked with mass wrapping across all edges. Moments
                    // must be taken about wrapped coordinates (peak at the origin), not
                    // raw indices, otherwise the variance is biased low.
                    double vx = (x > W / 2) ? (double)x - (double)W : (double)x;
                    double vy = (y > Hh / 2) ? (double)y - (double)Hh : (double)y;
                    double vz = (z > D / 2) ? (double)z - (double)D : (double)z;
                    total += v; mx += v * vx; my += v * vy; mz += v * vz;
                    mx2 += v * vx * vx; my2 += v * vy * vy; mz2 += v * vz * vz;
                }
            }
    hostMem.freeMemoryOnDevice(psfGhost);

    auto sd = [total](double m1, double m2) {
        if (total <= 0.0) return 0.0;
        double var = m2 / total - (m1 / total) * (m1 / total);
        return var > 0.0 ? std::sqrt(var) : 0.0;
    };
    sigX = std::max(0.1f, static_cast<real_t>(sd(mx, mx2)));
    sigY = std::max(0.1f, static_cast<real_t>(sd(my, my2)));
    sigZ = std::max(0.1f, static_cast<real_t>(sd(mz, mz2)));
}



} // namespace

void PoissonMLEDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Accept both PoissonMLEConfig and PoissonMLEUnitStepConfig (the latter is
    // derived from the former).
    configure(asConfig<PoissonMLEConfig>(config, DeconvolutionAlgorithms::kPoissonMLE));
}

void PoissonMLEDeconvolutionAlgorithm::configure(const PoissonMLEConfig& config) {
    iterations = config.iterations;
    signal_to_noise = config.signalToNoise;
    lambdaOverride = config.lambda;
    backgroundOverride = config.background;
    usePrefilter = config.prefilter;
    prefilterScale = config.prefilterScale;
    useSieve = config.sieve;
    usePrior = config.prior;
    sievePasses = config.sievePasses;
    sieveScale = config.sieveScale;
}

void PoissonMLEDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE algorithm initialization");
    const IBackendMemoryManager& memory = backend->getMemoryManager();

    // One real in-place-layout scratch buffer (m -> d per iteration); its
    // half-complex view (same memory) is the FFT work / result buffer. The
    // model m and the search direction d are never needed simultaneously in
    // the unit-step path, so they share this buffer.
    scratch = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    scratchC = memory.reinterpret(scratch);

    // Pre-register the real r2c/c2r plans (in-place); others are created lazily.
    FFTPlanDescription fReal{PlanDirection::FORWARD, PlanType::REAL, dataSize, true};
    FFTPlanDescription bReal{PlanDirection::BACKWARD, PlanType::REAL, dataSize, true};
    auto& plans = backend->mutableComputeManager();
    plans.initializePlan(fReal);
    plans.initializePlan(bReal);

    psfSigmasValid = false;
    initialized = true;
}

bool PoissonMLEDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void PoissonMLEDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f,
                                                 const DeconvolutionRunContext& runContext) {
    assert(backend && "No backend available for PoissonMLE algorithm");
    assert(initialized && "PoissonMLE algorithm not initialized. Call init() first.");
    if (iterations <= 0) return;

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& compute = backend->getComputeManager();
    const CuboidShape shape = f.getSize();

    assert(memory.isOnDevice(f) && "f is not on device");
    assert(memory.isOnDevice(g) && "g is not on device");


    // ---- Initial estimate: the data, optionally prefiltered. ----
    memory.memCopy(g, f);

    // ---- Physical parameters: the PSF is the same for every cube; the
    // padded shape is run-constant, so the estimate is cached after the first
    // cube. Re-estimating per cube would let float32 OTF noise give slightly
    // different prefilter/sieve widths and thus visible tile seams.
    if (!psfSigmasValid) {
        estimatePsfSigmas(compute, memory, H, shape, scratchC, scratch, psfSigX, psfSigY, psfSigZ);
        psfSigmasValid = true;
    }
    const real_t sigX = psfSigX;
    const real_t sigY = psfSigY;
    const real_t sigZ = psfSigZ;
    const real_t psfLat = 0.5f * (sigX + sigY);
    const real_t psfAx = sigZ;

    // ---- Prefilter: a single mild anisotropic Gaussian low-pass of the raw data. ----
    // sigma_axis = prefilterScale * PSF sigma_axis (0.3 * PSF width reproduces the
    // reference "sigmas of initial filt"). It gives the MLE a smooth, positive
    // starting object f0 (instead of raw photon-noise spikes).
    // f is already in the real-FFT in-place layout (allocated RealFFTInPlace by the
    // executor), so the in-place FFT low-pass can run on it directly.
    if (usePrefilter && prefilterScale > 0.0f) {
        applySieve(compute, f, scratchC,
                   prefilterScale * sigX, prefilterScale * sigY, prefilterScale * sigZ);
        compute.realClampMin(f, 0.0f, f);
        if (auto logger = spdlog::get("deconvolution"))
            logger->debug(
                "PoissonMLE prefilter: sigmas of initial filt: {:.3f} {:.3f} {:.3f}",
                prefilterScale * sigX, prefilterScale * sigY, prefilterScale * sigZ);
    }

    // Background: an explicit level (config) wins; otherwise the plan-provided
    // run-wide value (one value for every cube -> no tile seams). Without
    // either, fail: a per-tile estimate would differ between neighbouring
    // cubes and leave visible seams.
    real_t bg;
    if (backgroundOverride > -1.0f) {
        bg = backgroundOverride;
    } else if (runContext.background.has_value()) {
        bg = *runContext.background;
    } else {
        throw std::runtime_error("Missing background estimation for PoissonMLE deconvolution algorithm");
    }

    // SNR -> shrinkage strength lambda. 0 = derive from SNR (lambda = base/snr:
    // higher SNR -> weaker prior, closer to the pure MLE).
    const real_t lambda = (lambdaOverride > 0.0f)
                        ? lambdaOverride
                        : kLambdaBase / std::max(1e-6f, signal_to_noise);

    // f is the object ABOVE the background (m = Hf + bg), so the physical
    // constraint is f >= 0 and the background of f converges to 0 (where the
    // model m -> bg matches the observed background g -> bg). Clamping at bg
    // instead would pin the background of f at bg and make the model predict
    // 2*bg there — the background can never be fit.
    compute.realClampMin(f, 0.0f, f);

    const int maxIter = iterations;
    const real_t kLast = std::max(1.0f, 0.8f * static_cast<real_t>(maxIter));
    resetDataStepState(); // reset variant-specific data-step state (e.g. damping)

    for (int k = 0; k < maxIter; ++k) {
        // ---- DATA STEP: Poisson likelihood gradient + variant-specific step. ----
        applyDataStep(compute, H, g, bg, f);

        // ---- PRIOR STEP: lambda-shrinkage with flux conservation. ----
        if (usePrior && lambda > 1e-6f) {
            real_t fOld = 0.0f;
            compute.realSum(f, &fOld);
            compute.realSqrtAffine(f, 2.0f * lambda, 1.0f, lambda, f); // (sqrt(2*lambda*f + 1) - 1)/lambda
            compute.realClampMin(f, 0.0f, f);
            real_t fNew = 0.0f;
            compute.realSum(f, &fNew);
            if (fOld > 1e-12f && fNew > 1e-12f) {
                compute.realLinearCombination(f, fOld / fNew, f, 0.0f, f); // conserve total flux
            }
        }

        // ---- SIEVE: annealed anisotropic Gaussian smoothing. ----
        // The smoothing width carries two independent dials: the SNR (how much
        // regularizing the data warrants) and the annealing schedule (rough early,
        // fine late). The prior strength is lambda = base/snr; the sieve mirrors
        // this so signal_to_noise stays the master dial for BOTH regularizers:
        // effective scale = poissonSieveScale / snr, i.e. the sieve is as wide as
        // poissonSieveScale at SNR=1 and fades toward the identity as SNR -> inf
        // (the pure-MLE limit), while at low SNR it blurs strongly to suppress the
        // noise the MLE would otherwise amplify.
        const real_t sieveSnrScale = sieveScale / std::max(1e-6f, signal_to_noise);
        if (useSieve && sieveSnrScale > 1e-6f) {
            const real_t t = std::max(0.0f, 1.0f - static_cast<real_t>(k) / kLast);
            const real_t sl = sieveSnrScale * psfLat * t;
            const real_t sa = sieveSnrScale * psfAx * t;
            for (int p = 0; p < sievePasses; ++p) {
                applySieve(compute, f, scratchC, sl, sl, sa);
            }
        }

        compute.realClampMin(f, 0.0f, f);
        backend->sync();
        if (progressFunction) progressFunction(maxIter);
    }

}

void PoissonMLEDeconvolutionAlgorithm::computeForwardModel(
    const IComputeBackend& compute, const ComplexData& H, const RealData& f,
    real_t bg, RealData& m) {
    const IBackendMemoryManager& memory = backend->getMemoryManager();
    // m = Hf + bg  (r2c/c2r are always run in-place on a buffer's real/complex pair,
    // exactly as in RichardsonLucy, because the out-of-place real r2c plan assumes
    // the padded in-place layout. So the FFT source is first copied into the buffer.)
    memory.memCopy(f, m);                         // m <- f
    ComplexView mC = m.reinterpret();             // non-owning complex view of m
    compute.forwardFFT(m, mC);                    // in-place r2c -> mC = FFT(f)
    compute.complexMultiplication(mC, H, mC);
    compute.backwardFFT(mC, m);                   // in-place c2r -> m = Hf
    compute.realAffine(m, 1.0f, bg, m);           // m = Hf + bg
    compute.realClampMin(m, bg, m);               // physical floor (see header)
}

void PoissonMLEDeconvolutionAlgorithm::computeSearchDirection(
    const IComputeBackend& compute, const ComplexData& H, const RealData& g,
    const RealData& f, RealData& m, RealData& d) {
    // ratio R = H^T(g / m)  (clobbers m when d is the same buffer)
    compute.division(g, m, d, 1e-9f);             // d = g/m
    ComplexView dC = d.reinterpret();             // non-owning complex view of d
    compute.forwardFFT(d, dC);                    // in-place r2c -> dC
    compute.complexMultiplicationWithConjugate(dC, H, dC);
    compute.backwardFFT(dC, d);                   // in-place c2r -> d = R

    // search direction d = -f * (1 - R)
    compute.realAffine(d, -1.0f, 1.0f, d);        // 1 - R
    compute.multiplication(f, d, d);              // f*(1-R)
    compute.realAffine(d, -1.0f, 0.0f, d);        // -f*(1-R) = d
}

void PoissonMLEDeconvolutionAlgorithm::applyDataStep(
    const IComputeBackend& compute, const ComplexData& H, const RealData& g,
    real_t bg, RealData& f) {
    // m = Hf + bg, then d = -f*(1 - H^T(g/m)), both in the shared `scratch`:
    // the unit step never needs m past the g/m ratio, so m is allowed to be
    // clobbered by the direction computation.
    computeForwardModel(compute, H, f, bg, scratch);
    computeSearchDirection(compute, H, g, f, scratch, scratch);

    // Plain RL unit step: f <- f + d, clipped at zero. No hd = H*d forward model
    // and no line search, so this iteration is much cheaper than the adaptive
    // subclass's. This is the multiplicative RL update written additively, and it
    // provably does not increase the Poisson (I-divergence) objective.
    compute.realLinearCombination(f, 1.0f, scratch, 1.0f, f);
    compute.realClampMin(f, 0.0f, f);
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEDeconvolutionAlgorithm>();
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

size_t PoissonMLEDeconvolutionAlgorithm::getWorkingSetBytes(const CuboidShape& dataSize) const {
    // Allocates 1 additional array of input size
    // (scratch as real in-place-FFT buffer)
    return dataSize.getVolume() * sizeof(real_t);
}
