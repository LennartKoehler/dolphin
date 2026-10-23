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

const real_t kAlphaLow = 0.005f;
const real_t kAlphaHigh = 25.0f;
// Shrinkage base: lambda = kLambdaBase / snr (0 = derive from SNR). Higher SNR
// -> smaller lambda -> weaker prior (closer to the pure MLE).
const real_t kLambdaBase = 1.0f;

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
    mem.memCopy(H.getData(), work.getData(), H.getDataBytes(), H.getSize());
    compute.backwardFFT(work, workReal);

    const size_t W = shape.width, Hh = shape.height, D = shape.depth;

    // Copy the recovered (device) PSF to the host before the moment loop — the
    // host pointer must never be dereferenced while it still points at device
    // memory (CUDA). Same copy-to-host idiom as estimateBackground; the host copy
    // preserves the padded in-place layout, so the raw stride math below holds.
    const IBackendMemoryManager& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();
    RealData psfCopy = mem.copyDataToDevice(workReal);
    RealData psfGhost = mem.moveDataFromDevice(psfCopy, hostMem);
    const real_t* d = psfGhost.getData();
    const size_t stride = 2 * (W / 2 + 1);
    const size_t sliceStride = stride * Hh;

    double total = 0, mx = 0, my = 0, mz = 0, mx2 = 0, my2 = 0, mz2 = 0;
    for (size_t z = 0; z < D; ++z)
        for (size_t y = 0; y < Hh; ++y)
            for (size_t x = 0; x < W; ++x) {
                real_t v = d[z * sliceStride + y * stride + x];
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

// Background = mean of the dimmest decile of the (host-copied) image.
real_t estimateBackground(const RealData& g, const IBackendMemoryManager& mem) {
    const IBackendMemoryManager& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();
    RealData gcopy = mem.copyDataToDevice(g);
    RealData ghost = mem.moveDataFromDevice(gcopy, hostMem);
    const size_t n = ghost.getSize().getVolume();
    std::vector<real_t> vals;
    vals.reserve(n);
    for (size_t i = 0; i < n; ++i) vals.push_back(ghost.access(i));
    hostMem.freeMemoryOnDevice(ghost);
    if (vals.empty()) return 0.0f;
    std::sort(vals.begin(), vals.end());
    size_t lo = std::max<size_t>(1, n * 10 / 100); // dimmest decile
    if (lo > n) lo = n;
    double mean = 0;
    for (size_t i = 0; i < lo; ++i) mean += vals[i];
    return static_cast<real_t>(mean / lo);
}

} // namespace

void PoissonMLEDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    iterations = config.iterations;
    snr = config.poissonSnr;
    lambdaOverride = config.poissonLambda;
    backgroundOverride = config.poissonBackground;
    useSieve = config.poissonSieve;
    usePrior = config.poissonPrior;
    sievePasses = config.poissonSievePasses;
}

void PoissonMLEDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE algorithm initialization");
    const IBackendMemoryManager& memory = backend->getMemoryManager();

    // Real in-place-layout scratch buffers; each also exposes a half-complex view
    // (same memory) used as the FFT work / result buffer.
    model = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);     // m = Hf + bg
    modelC = memory.reinterpret(model);
    direction = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize); // search direction d
    directionC = memory.reinterpret(direction);
    hd = memory.allocateMemoryOnDeviceRealFFTInPlace(dataSize);        // H*d for the line search
    hdC = memory.reinterpret(hd);
    hNorm = memory.allocateMemoryOnDeviceComplex(dataSize);            // unit-sum PSF OTF
    sieveOTF = memory.allocateMemoryOnDeviceComplex(dataSize);         // half-complex Gaussian low-pass

    // Pre-register the real r2c/c2r plans (in-place); others are created lazily.
    FFTPlanDescription fReal{PlanDirection::FORWARD, PlanType::REAL, dataSize, true};
    FFTPlanDescription bReal{PlanDirection::BACKWARD, PlanType::REAL, dataSize, true};
    auto& plans = backend->mutableComputeManager();
    plans.initializePlan(fReal);
    plans.initializePlan(bReal);

    initialized = true;
}

bool PoissonMLEDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void PoissonMLEDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f) {
    assert(backend && "No backend available for PoissonMLE algorithm");
    assert(initialized && "PoissonMLE algorithm not initialized. Call init() first.");
    if (iterations <= 0) return;

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& compute = backend->getComputeManager();
    const CuboidShape shape = f.getSize();

    assert(memory.isOnDevice(f.getData()) && "f is not on device");
    assert(memory.isOnDevice(g.getData()) && "g is not on device");

    // ---- PSF normalization: unit sum. ----
    // The forward model m = Hf + bg (and the photon flux bookkeeping) is only
    // physically correct when the OTF's DC term (= sum of the PSF) equals 1.
    // RichardsonLucy is scale-invariant and never needed this; the I-divergence
    // is not. The backend reads the DC bin per its memory model and scales the
    // buffer so its DC term is 1.
    memory.memCopy(H, hNorm);
    compute.normalizePsfUnitSum(hNorm, 1e-9f);

    // ---- Initial estimate: the data itself. ----
    memory.memCopy(g, f);

    // ---- Physical parameters, estimated per brick. ----
    real_t sigX = 1.0f, sigY = 1.0f, sigZ = 1.0f;
    estimatePsfSigmas(compute, memory, hNorm, shape, modelC, model, sigX, sigY, sigZ);
    const real_t psfLat = 0.5f * (sigX + sigY);
    const real_t psfAx = sigZ;

    const real_t bg = (backgroundOverride > -1.0f) ? backgroundOverride : estimateBackground(g, memory);

    // SNR -> shrinkage strength lambda. 0 = derive from SNR (lambda = base/snr:
    // higher SNR -> weaker prior, closer to the pure MLE).
    const real_t lambda = (lambdaOverride > 0.0f)
                        ? lambdaOverride
                        : kLambdaBase / std::max(1e-6f, snr);

    // f is the object ABOVE the background (m = Hf + bg), so the physical
    // constraint is f >= 0 and the background of f converges to 0 (where the
    // model m -> bg matches the observed background g -> bg). Clamping at bg
    // instead would pin the background of f at bg and make the model predict
    // 2*bg there — the background can never be fit.
    compute.realClampMin(f, 0.0f, f);

    const int maxIter = iterations;
    const real_t kLast = std::max(1.0f, 0.8f * static_cast<real_t>(maxIter));
    real_t pwr = 1.0f;

    for (int k = 0; k < maxIter; ++k) {
        // ---- DATA STEP: Poisson likelihood, gradient + exact Newton line search. ----
        // (r2c/c2r are always run in-place on a buffer's real/complex pair, exactly as
        //  in RichardsonLucy, because the out-of-place real r2c plan assumes the padded
        //  in-place layout. So each FFT source is first copied into its work buffer.)
        // m = Hf + bg  (spatial result in `model`)
        memory.memCopy(f, model);                     // model <- f
        compute.forwardFFT(model, modelC);            // in-place r2c -> modelC = FFT(f)
        compute.complexMultiplication(modelC, hNorm, modelC);
        compute.backwardFFT(modelC, model);           // in-place c2r -> model = Hf
        compute.realAffine(model, 1.0f, bg, model);   // model = Hf + bg = m
        // Physical floor of the model. With f >= 0 and the (non-negative, unit-sum)
        // PSF, m = Hf + bg >= bg exactly. But the sieve's float32 ringing can push f
        // slightly negative (and a measured PSF can have small negative lobes), dragging
        // m to <= 0 at isolated voxels. Left alone, the ratio g/m below is floored at the
        // tiny epsilon (1e-9) and amplifies g by ~1e9, running the iteration away to
        // inf/NaN. Clamp m to its physical lower bound bg so g/m stays <= g_max/bg and
        // the step remains bounded.
        compute.realClampMin(model, bg, model);

        // ratio R = H^T(g / m)  (spatial result in `direction`)
        compute.division(g, model, direction, 1e-9f); // direction = g/m
        compute.forwardFFT(direction, directionC);    // in-place r2c -> directionC
        compute.complexMultiplicationWithConjugate(directionC, hNorm, directionC);
        compute.backwardFFT(directionC, direction);   // in-place c2r -> direction = R

        // search direction d = -f * (1 - R)  (in `direction`)
        compute.realAffine(direction, -1.0f, 1.0f, direction); // 1 - R
        compute.multiplication(f, direction, direction);       // f*(1-R)
        compute.realAffine(direction, -1.0f, 0.0f, direction); // -f*(1-R) = d

        // hd = H*d  (spatial result in `hd`)
        memory.memCopy(direction, hd);               // hd <- d
        compute.forwardFFT(hd, hdC);                 // in-place r2c -> hdC = FFT(d)
        compute.complexMultiplication(hdC, hNorm, hdC);
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
        // f is the object above the background; keep it non-negative so the
        // background converges to 0 and the model m = Hf + bg matches g there.
        compute.realLinearCombination(f, 1.0f, direction, alpha * pwr, f);
        compute.realClampMin(f, 0.0f, f);

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
        if (useSieve) {
            const real_t t = std::max(0.0f, 1.0f - static_cast<real_t>(k) / kLast);
            const real_t sl = 0.5f * psfLat * t;
            const real_t sa = 0.5f * psfAx * t;
            for (int p = 0; p < sievePasses; ++p) {
                applySieve(compute, f, sieveOTF, sl, sl, sa);
            }
        }

        backend->sync();
        if (progressFunction) progressFunction(maxIter);
    }

    // ---- FINAL: remove the residual flat floor. ----
    // m = Hf + bg cannot distinguish a uniform offset in f from bg itself (H
    // preserves constants), so the MLE splits the difference: it absorbs a flat
    // residual equal to (background mean of g) - bg into f. The dimmest decile
    // estimate of bg is below the background mean, so the output background
    // settles at that difference instead of 0. The floor is uniform, so one
    // global subtraction (the dimmest decile of the result, i.e. the background)
    // removes it and leaves the object on a zero background.
    const real_t floor = estimateBackground(f, memory);
    if (floor > 0.0f) {
        compute.realAffine(f, 1.0f, -floor, f); // f -= floor
        compute.realClampMin(f, 0.0f, f);
    }
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEDeconvolutionAlgorithm>();
    copy->iterations = iterations;
    copy->snr = snr;
    copy->lambdaOverride = lambdaOverride;
    copy->backgroundOverride = backgroundOverride;
    copy->useSieve = useSieve;
    copy->usePrior = usePrior;
    copy->sievePasses = sievePasses;
    copy->initialized = false;
    return copy;
}

size_t PoissonMLEDeconvolutionAlgorithm::getMemoryMultiplier() const {
    return 5; // model, direction, hd (real, each a complex buffer's real view) + hNorm + sieveOTF (half-complex)
}
