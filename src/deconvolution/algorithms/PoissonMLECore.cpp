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
// the standard deviation of its positive mass. The moment reduction runs entirely
// on the backend (realWrappedMoments): only the 7 scalar moments cross to the
// host — no volume copy, no device temp buffer.
void estimatePsfSigmas(const IComputeBackend& compute,
                       const IBackendMemoryManager& mem,
                       const ComplexData& H,
                       ComplexData& work, RealData& workReal,
                       real_t& sigX, real_t& sigY, real_t& sigZ) {
    // work = H (copy), then in-place c2r -> spatial PSF in workReal.
    mem.memCopy(H, work);
    compute.backwardFFT(work, workReal);

    // One device-side pass: wrapped-coordinate moments of the PSF's positive
    // mass. The OTF comes from an octant-shifted PSF, so the recovered spatial
    // PSF is corner-peaked with mass wrapping across all edges — moments about
    // raw indices would bias the variance low, hence the wrapped coordinates
    // inside the kernel.
    double m[7];
    compute.realWrappedMoments(workReal, m);
    const double total = m[0];

    auto sd = [total](double m1, double m2) {
        if (total <= 0.0) return 0.0;
        double var = m2 / total - (m1 / total) * (m1 / total);
        return var > 0.0 ? std::sqrt(var) : 0.0;
    };
    sigX = std::max(0.1f, static_cast<real_t>(sd(m[1], m[4])));
    sigY = std::max(0.1f, static_cast<real_t>(sd(m[2], m[5])));
    sigZ = std::max(0.1f, static_cast<real_t>(sd(m[3], m[6])));
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

std::vector<FFTPlanDescription> PoissonMLECore::planDescriptions(const CuboidShape& dataSize) {
    return {
        // In-place r2c for the ratio FFT and the sieve, in-place c2r for both,
        // plus the out-of-place forward r2c the forward model uses to FFT the
        // iterate directly into the work buffer (the RichardsonLucy pattern).
        FFTPlanDescription{PlanDirection::FORWARD, PlanType::REAL, dataSize, true},
        FFTPlanDescription{PlanDirection::BACKWARD, PlanType::REAL, dataSize, true},
        FFTPlanDescription{PlanDirection::FORWARD, PlanType::REAL, dataSize, false},
    };
}

void PoissonMLECore::initPlans(IComputeBackend& plans, const CuboidShape& dataSize) const {
    for (const FFTPlanDescription& description : planDescriptions(dataSize)) {
        plans.initializePlan(description);
    }
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
                                     const ComplexData& H,
                                     ComplexData& workOTF, RealData& work) {
    if (psfSigmasValid) return;
    estimatePsfSigmas(compute, memory, H, workOTF, work, psfSigX, psfSigY, psfSigZ);
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
