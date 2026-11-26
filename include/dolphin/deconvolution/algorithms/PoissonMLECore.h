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

#pragma once

#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"
#include "dolphinbackend/ComplexData.h"

struct DeconvolutionRunContext;
struct CuboidShape;
struct FFTPlanDescription;
class IComputeBackend;
class IBackendMemoryManager;

// State + shared math of the Poisson/I-divergence (photon-counting) MLE
// deconvolution family, shared by composition (NOT inheritance) by the
// concrete variants:
//   PoissonMLE           - RL unit step, classic in-core data step
//   PoissonMLEStreamed   - RL unit step, out-of-core (g streamed) data step
//   PoissonMLEAdaptiveStep - exact Newton line search data step
//
// Owns everything the variants share: the mirrored PoissonMLEConfig fields,
// the cached PSF width estimate, and the classic work buffer. The backend
// interfaces are passed into the methods (the core holds no backend), so the
// variants keep ownership of setup and iteration flow.
//
// Each iteration of every variant is the same alternating scheme on the
// objective J(f) = sum [ g ln(g/m) - g + m ],  m = Hf + bg,  f >= 0:
//   DATA STEP       variant-specific (unit step / streamed / Newton line search)
//   PRIOR STEP      lambda-shrinkage f <- (sqrt(2*lambda*f + 1) - 1)/lambda
//                   + flux renormalization
//   SIEVE           annealed anisotropic Gaussian smoothing (roughness regularizer)
// Stopping is by iteration count (no quality-factor criterion).
struct PoissonMLECore {
    // ---- Configuration mirrored from PoissonMLEConfig. ----
    int iterations = 10;
    float signal_to_noise = 10.0f;
    float lambdaOverride = 0.0f;
    float backgroundOverride = -1.0f;
    bool usePrefilter = false;
    float prefilterScale = 0.0f;
    bool useSieve = false;
    bool usePrior = false;
    int sievePasses = 1;
    float sieveScale = 1.0f;

    // ---- Cached PSF width estimate (run-constant for a given padded shape). ----
    bool psfSigmasValid = false;
    real_t psfSigX = 1.0f, psfSigY = 1.0f, psfSigZ = 1.0f;
    // Derived sieve widths: lateral = mean of X/Y sigma, axial = Z sigma.
    real_t psfSigLat = 1.0f, psfSigAx = 1.0f;

    // ---- Classic work buffer: one real in-place-FFT buffer (m -> ratio per
    // iteration); its half-complex view (same memory) is the FFT work / result
    // buffer and the sieve's Gaussian-OTF buffer. Not used by the streamed
    // variant (its work volume is the caller's g buffer; it keeps only a
    // one-slice staging buffer instead). ----
    RealData scratch;
    ComplexView scratchC;

    // Copyable for clone(): copies configuration and the cached PSF widths,
    // deliberately NOT the device buffers (scratch/scratchC). A clone is
    // re-initialized and re-allocated by the caller (init/preAllocate run on
    // every clone), and DataView itself is non-copyable, so the buffers must
    // never be carried over.
    PoissonMLECore() = default;
    PoissonMLECore(const PoissonMLECore& other)
        : iterations(other.iterations),
          signal_to_noise(other.signal_to_noise),
          lambdaOverride(other.lambdaOverride),
          backgroundOverride(other.backgroundOverride),
          usePrefilter(other.usePrefilter),
          prefilterScale(other.prefilterScale),
          useSieve(other.useSieve),
          usePrior(other.usePrior),
          sievePasses(other.sievePasses),
          sieveScale(other.sieveScale),
          psfSigmasValid(other.psfSigmasValid),
          psfSigX(other.psfSigX), psfSigY(other.psfSigY), psfSigZ(other.psfSigZ),
          psfSigLat(other.psfSigLat), psfSigAx(other.psfSigAx) {}
    PoissonMLECore& operator=(const PoissonMLECore& other) {
        if (this == &other) return *this;
        iterations = other.iterations;
        signal_to_noise = other.signal_to_noise;
        lambdaOverride = other.lambdaOverride;
        backgroundOverride = other.backgroundOverride;
        usePrefilter = other.usePrefilter;
        prefilterScale = other.prefilterScale;
        useSieve = other.useSieve;
        usePrior = other.usePrior;
        sievePasses = other.sievePasses;
        sieveScale = other.sieveScale;
        psfSigmasValid = other.psfSigmasValid;
        psfSigX = other.psfSigX; psfSigY = other.psfSigY; psfSigZ = other.psfSigZ;
        psfSigLat = other.psfSigLat; psfSigAx = other.psfSigAx;
        return *this;
    }

    void configure(const PoissonMLEConfig& config);
    // Invalidates per-shape cached state; call from cloneSpecific() (the
    // clone is re-initialized and re-allocated by the caller).
    void resetForClone() { psfSigmasValid = false; }

    // Prewarms the in-place real r2c/c2r plans both FFT stages of every variant
    // use. Kept in init (not preAllocate) because plan creation transiently
    // needs a work area, which the caller wants out of the way of the image
    // buffers.
    void initPlans(IComputeBackend& plans, const CuboidShape& dataSize) const;

    // Allocates the classic work buffer. The streamed variant does NOT call
    // this (its work volume is caller-provided).
    void allocateWork(const IBackendMemoryManager& memory, const CuboidShape& dataSize);

    // Background for the model m = Hf + bg: an explicit config level wins,
    // otherwise the plan-provided run-wide value (one value for every cube so
    // neighbouring cubes agree -> no tile seams). Throws when neither exists:
    // a per-tile estimate would differ between neighbouring cubes.
    real_t resolveBackground(const DeconvolutionRunContext& runContext) const;

    // Shrinkage strength lambda: explicit override wins, else base/SNR.
    real_t currentLambda() const;

    // Optional mild anisotropic Gaussian low-pass of the raw data (the MLE's
    // initial estimate), plus the debug log of the effective sigmas. workOTF
    // serves as the half-complex Gaussian-OTF buffer.
    void applyPrefilter(const IComputeBackend& compute, RealData& f,
                        ComplexData& workOTF, real_t sigX, real_t sigY, real_t sigZ) const;

    // Everything after the data step in one iteration: lambda-shrinkage prior
    // with flux conservation, the annealed sieve, and the physical clamp.
    void applyRegularization(const IComputeBackend& compute, RealData& f,
                             ComplexData& workOTF, int k, int kLast, real_t lambda);

    // Estimates (once, cached) the PSF's sigma widths using `work`/`workOTF` as
    // scratch. Run-constant for a given padded shape: re-estimating per cube
    // would let float32 OTF noise give slightly different prefilter/sieve
    // widths and thus visible tile seams.
    void ensurePsfSigmas(const IComputeBackend& compute, const IBackendMemoryManager& memory,
                         const ComplexData& H, const CuboidShape& shape,
                         ComplexData& workOTF, RealData& work);

    // m = clamp_min(Hf + bg, bg), written into `m`. The clamp is the model's
    // physical floor: with f >= 0 and the (non-negative, unit-sum) PSF,
    // m = Hf + bg >= bg exactly. But float32 ringing can push f slightly
    // negative (and a measured PSF can have small negative lobes), dragging m
    // to <= 0 at isolated voxels. Left alone, the g/m ratio is floored at the
    // tiny epsilon (1e-9) and amplifies g by ~1e9, running the iteration away
    // to inf/NaN.
    void computeForwardModel(const IComputeBackend& compute, const IBackendMemoryManager& memory,
                             const ComplexData& H, const RealData& f, real_t bg, RealData& m) const;

    // The shared core of both data steps: the ratio R = H^T(g/m), written into
    // `ratio` (m is clobbered when they alias).
    void computeRatio(const IComputeBackend& compute, const ComplexData& H,
                      const RealData& g, const RealData& m, RealData& ratio) const;

    // d = f*(R - 1) written into `d` (the line-search direction). Only the
    // adaptive variant uses this: it needs the direction d AND the model m
    // (for J(alpha)) alive simultaneously, so m must be a separate buffer.
    void computeSearchDirection(const IComputeBackend& compute, const ComplexData& H,
                                const RealData& g, const RealData& f,
                                RealData& m, RealData& d) const;
};