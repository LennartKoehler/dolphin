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

#include "dolphin/deconvolution/algorithms/PoissonMLECore.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h" // DeconvolutionRunContext
#include "dolphin/backend/BackendFactory.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace {

// Shrinkage base: lambda = kLambdaBase / snr (0 = derive from SNR). Higher SNR
// -> smaller lambda -> weaker prior (closer to the pure MLE).
const real_t kLambdaBase = 1.0f;

// Annealed anisotropic Gaussian smoothing in the frequency domain (the "sieve").
// Operates in-place on `f` (which has the in-place-FFT layout): r2c, multiply by
// the analytically evaluated Gaussian OTF (fused kernel, no OTF buffer), c2r.
// `f` is both the source and the output.
void applySieve(const IComputeBackend& compute, RealData& f,
                real_t sigmaX, real_t sigmaY, real_t sigmaZ) {
    if (sigmaX < 0.05f && sigmaY < 0.05f && sigmaZ < 0.05f) {
        return; // smoothing width ~ 0, sieve is the identity
    }
    ComplexView fv = f.reinterpret();       // non-owning complex view of f
    compute.forwardFFT(f, fv);              // in-place r2c -> f (frequency domain)
    compute.complexMultiplyGaussianOTF(fv, sigmaX, sigmaY, sigmaZ);
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

void PoissonMLECore::configure(const PoissonMLEConfig& config) {
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

void PoissonMLECore::initPlans(IComputeBackend& plans, const CuboidShape& dataSize) const {
    // In-place plans for the ratio FFT and the sieve (both run in-place on a
    // buffer's real/complex pair), plus the out-of-place forward plan the
    // forward model uses to FFT the iterate directly into the work buffer
    // (same pattern as RichardsonLucy: out-of-place r2c from a padded
    // in-place-layout source).
    FFTPlanDescription fReal{PlanDirection::FORWARD, PlanType::REAL, dataSize, true};
    FFTPlanDescription bReal{PlanDirection::BACKWARD, PlanType::REAL, dataSize, true};
    FFTPlanDescription fRealOOP{PlanDirection::FORWARD, PlanType::REAL, dataSize, false};
    plans.initializePlan(fReal);
    plans.initializePlan(bReal);
    plans.initializePlan(fRealOOP);
}

void PoissonMLECore::allocateWork(const IBackendMemoryManager& memory, const CuboidShape& dataSize) {
    // One real in-place-layout scratch buffer (m -> ratio per iteration); its
    // half-complex view (same memory) is the FFT work / result buffer. Also
    // serves as the sieve's Gaussian-OTF buffer.
    scratch = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    scratchC = memory.reinterpret(scratch);
}

real_t PoissonMLECore::resolveBackground(const DeconvolutionRunContext& runContext) const {
    if (backgroundOverride > -1.0f) {
        return backgroundOverride;
    }
    if (runContext.background.has_value()) {
        return *runContext.background;
    }
    throw std::runtime_error("Missing background estimation for PoissonMLE deconvolution algorithm");
}

real_t PoissonMLECore::currentLambda() const {
    // SNR -> shrinkage strength lambda. 0 = derive from SNR (lambda = base/snr:
    // higher SNR -> weaker prior, closer to the pure MLE).
    return (lambdaOverride > 0.0f)
         ? lambdaOverride
         : kLambdaBase / std::max(1e-6f, signal_to_noise);
}

void PoissonMLECore::applyPrefilter(const IComputeBackend& compute, RealData& f,
                                    real_t sigX, real_t sigY, real_t sigZ) const {
    // sigma_axis = prefilterScale * PSF sigma_axis (0.3 * PSF width reproduces the
    // reference "sigmas of initial filt"). It gives the MLE a smooth, positive
    // starting object f0 (instead of raw photon-noise spikes).
    // f is already in the real-FFT in-place layout (allocated RealFFTInPlace by the
    // executor), so the in-place FFT low-pass can run on it directly.
    if (!usePrefilter || prefilterScale <= 0.0f) return;

    applySieve(compute, f,
               prefilterScale * sigX, prefilterScale * sigY, prefilterScale * sigZ);
    compute.realClampMin(f, 0.0f, f);
    if (auto logger = spdlog::get("deconvolution"))
        logger->debug(
            "PoissonMLE prefilter: sigmas of initial filt: {:.3f} {:.3f} {:.3f}",
            prefilterScale * sigX, prefilterScale * sigY, prefilterScale * sigZ);
}

void PoissonMLECore::ensurePsfSigmas(const IComputeBackend& compute,
                                     const IBackendMemoryManager& memory,
                                     const ComplexData& H, const CuboidShape& shape,
                                     ComplexData& workOTF, RealData& work) {
    if (psfSigmasValid) return;
    estimatePsfSigmas(compute, memory, H, shape, workOTF, work, psfSigX, psfSigY, psfSigZ);
    psfSigLat = 0.5f * (psfSigX + psfSigY);
    psfSigAx = psfSigZ;
    psfSigmasValid = true;
}

void PoissonMLECore::applyRegularization(const IComputeBackend& compute, RealData& f,
                                         int k, int kLast, real_t lambda) {
    // ---- PRIOR STEP: lambda-shrinkage with flux conservation. ----
    if (usePrior && lambda > 1e-6f) {
        real_t fOld = 0.0f;
        compute.realSum(f, &fOld);
        // (sqrt(2*lambda*f + 1) - 1)/lambda, clamped at 0 in the same pass.
        compute.realSqrtAffineClampMin(f, 2.0f * lambda, 1.0f, lambda, 0.0f, f);
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
        const real_t sl = sieveSnrScale * psfSigLat * t;
        const real_t sa = sieveSnrScale * psfSigAx * t;
        for (int p = 0; p < sievePasses; ++p) {
            applySieve(compute, f, sl, sl, sa);
        }
    }

    compute.realClampMin(f, 0.0f, f);
}

void PoissonMLECore::computeForwardModel(
    const IComputeBackend& compute, const ComplexData& H, const RealData& f,
    real_t bg, RealData& m) const {
    // m = Hf + bg. The r2c runs OUT-OF-PLACE from the padded in-place-layout
    // iterate f directly into m's complex view (the plan is prewarmed with the
    // padded source layout, exactly as in RichardsonLucy) — no full-volume
    // copy of f into m per iteration.
    ComplexView mC = m.reinterpret();             // non-owning complex view of m
    compute.forwardFFT(f, mC);                    // out-of-place r2c -> mC = FFT(f)
    compute.complexMultiplication(mC, H, mC);
    compute.backwardFFT(mC, m);                   // in-place c2r -> m = Hf
    compute.realAffineClampMin(m, 1.0f, bg, bg, m); // m = max(Hf + bg, bg): model + physical floor in one pass
}

void PoissonMLECore::computeRatio(const IComputeBackend& compute, const ComplexData& H,
                                  const RealData& g, const RealData& m, RealData& ratio) const {
    // ratio R = H^T(g / m)  (clobbers m when ratio is the same buffer)
    compute.division(g, m, ratio, 1e-9f);             // ratio = g/m
    ComplexView ratioC = ratio.reinterpret();         // non-owning complex view of ratio
    compute.forwardFFT(ratio, ratioC);                // in-place r2c -> ratioC
    compute.complexMultiplicationWithConjugate(ratioC, H, ratioC);
    compute.backwardFFT(ratioC, ratio);               // in-place c2r -> ratio = R
}

void PoissonMLECore::computeSearchDirection(const IComputeBackend& compute, const ComplexData& H,
                                            const RealData& g, const RealData& f,
                                            RealData& m, RealData& d) const {
    computeRatio(compute, H, g, m, d);

    // search direction d = -f * (1 - R) = f * (R - 1), in one fused pass
    // (d aliases the ratio buffer: read before write within the pass).
    compute.realAffineProduct(f, d, 1.0f, -1.0f, d);
}