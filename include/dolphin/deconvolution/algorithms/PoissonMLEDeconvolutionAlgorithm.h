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

#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"
#include <memory>

// Poisson / I-divergence (photon-counting) likelihood deconvolution: an alternating
// data/prior scheme on the objective
//     J(f) = sum [ g ln(g/m) - g + m ],   m = Hf + bg,   f >= bg.
// Before the loop, an optional prefilter (a single mild anisotropic Gaussian
// low-pass of the raw data, sigma = prefilterScale * PSF sigma) sets the initial
// estimate f0. The background bg is resolved once for the whole image by the
// plan (DeconvolutionRunContext, background_strategy = global or whole_image)
// so that neighbouring cubes agree; deconvolving without it is an error
// (an explicit poisson_background configures it directly).
// Each iteration:
//   DATA STEP  - gradient H^T(1 - g/m), direction d = -f*(1 - ratio), then a step
//                f <- f + step*d clipped at zero. The default (this class) takes
//                the plain Richardson-Lucy unit step (step = 1), skipping the
//                hd = H*d FFT + line search entirely;
//                PoissonMLEAdaptiveStepDeconvolutionAlgorithm overrides the data
//                step with an exact Newton line search (guarded against the RL
//                unit step) at the cost of one extra FFT + ~30 J(alpha) evaluations
//                per iteration.
//   GUARD      - if J increased, damp the update (power) and redo the iteration
//                (adaptive-step variant only).
//   PRIOR STEP - lambda-shrinkage f <- (sqrt(2*lambda*f + 1) - 1)/lambda + flux renormalize.
//   SIEVE      - annealed anisotropic Gaussian smoothing (roughness regularizer).
// Stopping is by iteration count (no quality-factor criterion).
class PoissonMLEDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    PoissonMLEDeconvolutionAlgorithm() = default;
    ~PoissonMLEDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    virtual void configure(const PoissonMLEConfig& config);
    virtual void init(const CuboidShape& dataSize) override;
    virtual void preAllocate(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    virtual size_t getWorkingSetBytes(const CuboidShape& dataSize) const override;

protected:
    // Apply one data step: compute the model m = Hf + bg, the search direction
    // d = -f*(1 - H^T(g/m)), and move f. The default (this class) takes the plain
    // RL unit step f <- f + d, clipped at zero, reusing the shared `scratch` for
    // m and d (they are never needed simultaneously here). Subclasses can override
    // to use a different step rule (e.g. the exact Newton line search of
    // PoissonMLEAdaptiveStepDeconvolutionAlgorithm, which keeps m in its own buffer
    // and adds one extra hd = H*d FFT + J(alpha) evaluations per iteration).
    virtual void applyDataStep(const IComputeBackend& compute,
                               const ComplexData& H, const RealData& g,
                               real_t bg, RealData& f);

    // Per-deconvolution data-step state reset, called once before the iteration
    // loop. Subclasses with damping state (line-search variant) override it.
    virtual void resetDataStepState() {}

    // Whether the base deconvolve() loop may replace applyDataStep() with the
    // fused, g-streamed data step (IBackend::poissonDataStepStreamed), which
    // needs no algorithm-owned work volume. The line-search variant overrides
    // this to false: it needs m and the direction d alive simultaneously.
    virtual bool useStreamedDataStep() const { return true; }

    // m = clamp_min(Hf + bg, bg), written into `m` (in-place FFTs follow
    // RichardsonLucy: out-of-place real->complex, then in-place complex->real into
    // the same buffer's real view, so a real result is never clobbered by an FFT).
    // The clamp is the model's physical floor: with f >= 0 and the (non-negative,
    // unit-sum) PSF, m = Hf + bg >= bg exactly. But the sieve's float32 ringing can
    // push f slightly negative (and a measured PSF can have small negative lobes),
    // dragging m to <= 0 at isolated voxels. Left alone, the ratio g/m below is
    // floored at the tiny epsilon (1e-9) and amplifies g by ~1e9, running the
    // iteration away to inf/NaN.
    void computeForwardModel(const IComputeBackend& compute, const ComplexData& H,
                             const RealData& f, real_t bg, RealData& m);

    // d = -f*(1 - H^T(g/m)) written into `d`. When `m` and `d` are the same buffer
    // (unit-step path), m is clobbered by the g/m ratio — pass a separate buffer
    // for m to keep it alive (line-search path needs it for J(alpha)).
    void computeSearchDirection(const IComputeBackend& compute, const ComplexData& H,
                                const RealData& g, const RealData& f,
                                RealData& m, RealData& d);

    // The shared core of computeSearchDirection: the ratio R = H^T(g/m), written
    // into `ratio` (m is clobbered when they alias). Separated so the unit-step
    // path can consume R directly via poissonUpdate() without materialising the
    // direction d; the line-search path keeps using computeSearchDirection.
    void computeRatio(const IComputeBackend& compute, const ComplexData& H,
                      const RealData& g, const RealData& m, RealData& ratio);

    // Shared data-step scratch: one real in-place-FFT buffer (m -> d per iteration)
    // plus its complex view (same memory, FFT work buffer). Subclasses may reuse it.
    // Only used by the classic (non-streamed) data step — the line-search variant
    // and backends without poissonDataStepStreamed. Allocated by the subclass's
    // preAllocate (adaptive) or lazily in deconvolve (classic fallback).
    RealData scratch;
    ComplexView scratchC;

    // One z-slice of the padded in-place layout: staging buffer for the streamed
    // data step's slab-wise upload of g. Tiny (a 1/depth fraction of a volume).
    RealData ratioSlab;

    // Configuration mirrored from PoissonMLEConfig (protected so subclasses can
    // copy them in their own cloneSpecific()).
    int iterations = 10;
    float signal_to_noise = 10.0f;
    float lambdaOverride;
    float backgroundOverride;
    bool usePrefilter;
    float prefilterScale;
    bool useSieve;
    bool usePrior;
    int sievePasses;
    float sieveScale;

    // Cached PSF width estimate (run-constant for a given padded shape).
    bool psfSigmasValid = false;
    real_t psfSigX = 1.0f, psfSigY = 1.0f, psfSigZ = 1.0f;

    bool initialized = false;

private:

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};
