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
//                f <- f + step*d clipped at zero. The base class picks `step` with
//                an exact Newton line search (guarded against the RL unit step);
//                PoissonMLEUnitStepDeconvolutionAlgorithm uses the plain RL step
//                (step = 1) and skips the hd = H*d FFT + line search entirely.
//   GUARD      - if J increased, damp the update (power) and redo the iteration.
//   PRIOR STEP - lambda-shrinkage f <- (sqrt(2*lambda*f + 1) - 1)/lambda + flux renormalize.
//   SIEVE      - annealed anisotropic Gaussian smoothing (roughness regularizer).
// Stopping is by iteration count (no quality-factor criterion).
class PoissonMLEDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    PoissonMLEDeconvolutionAlgorithm() = default;
    ~PoissonMLEDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    virtual void configure(const PoissonMLEConfig& config);
    void init(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    size_t getWorkingSetBytes(const CuboidShape& dataSize) const override;

protected:
    // Apply one data step: given the search direction `d` (in `direction`) and the
    // current estimate `f`, move f. The base implementation computes hd = H*d, finds
    // the optimal step length with a safeguarded Newton line search (guarded against
    // the RL unit step), damps the step (pwr) if the likelihood worsened, and applies
    // f <- f + alpha*pwr*d clipped at zero. Subclasses can override to use a different
    // step rule (e.g. the plain RL unit step, which skips the hd FFT + line search).
    virtual void applyDataStep(const IComputeBackend& compute,
                               const ComplexData& H, const RealData& g,
                               const RealData& model, RealData& direction, RealData& f);

    // Per-deconvolution step-damping state (line-search variant uses it).
    real_t pwr = 1.0f;

    // Configuration mirrored from PoissonMLEConfig (protected so the UnitStep
    // subclass can copy them in its own cloneSpecific()).
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
    // Scratch (device) buffers. Each spatial quantity (model m, direction d, hd=H*d)
    // is a real in-place-FFT buffer; its ComplexView (same memory, non-owning) is the
    // FFT work buffer. FFTs follow RichardsonLucy: out-of-place real->complex, then
    // in-place complex->real into the same buffer's real view, so a real result is
    // never clobbered by an FFT.
    RealData model;       // forward model m = Hf + bg (spatial)
    RealData direction;   // search direction d = -f*(1 - H^T(g/m)) (spatial)
    RealData hd;          // H*d, the forward model of the direction (spatial)
    ComplexView modelC;   // complex view of `model` (FFT work buffer)
    ComplexView directionC; // complex view of `direction`
    ComplexView hdC;      // complex view of `hd`
    ComplexData sieveOTF; // annealed anisotropic Gaussian low-pass kernel

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};
