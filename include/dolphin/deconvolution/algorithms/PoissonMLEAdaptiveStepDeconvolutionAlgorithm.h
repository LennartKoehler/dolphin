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

#include "dolphin/deconvolution/algorithms/PoissonMLEDeconvolutionAlgorithm.h"

// PoissonMLE with an exact Newton line search instead of the base class's plain
// Richardson-Lucy unit step. The data step computes the forward model of the
// direction, hd = H*d, then minimizes the Poisson objective J(alpha) along
// f + alpha*d (safeguarded against the RL unit step), damping the update power
// when the likelihood worsened. Because a step length is searched for, the model
// m must stay alive alongside the direction d, and each iteration costs one extra
// full-volume FFT plus ~30 J(alpha) evaluations on top of the unit-step base —
// traded for potentially fewer iterations. Prior and sieve steps are identical to
// the base class.
class PoissonMLEAdaptiveStepDeconvolutionAlgorithm : public PoissonMLEDeconvolutionAlgorithm {
public:
    PoissonMLEAdaptiveStepDeconvolutionAlgorithm() = default;
    ~PoissonMLEAdaptiveStepDeconvolutionAlgorithm() = default;

    // Unhide the base's configure(PoissonMLEConfig&) overload: declaring
    // configure(const DeconvolutionConfig&) below would otherwise hide it, and
    // the DeconvolutionConfig override's internal configure(asConfig<...>()) call
    // would resolve to itself (infinite recursion).
    using PoissonMLEDeconvolutionAlgorithm::configure;

    void configure(const DeconvolutionConfig& config) override;
    void init(const CuboidShape& dataSize) override;
    size_t getWorkingSetBytes(const CuboidShape& dataSize) const override;

protected:
    void applyDataStep(const IComputeBackend& compute, const ComplexData& H,
                       const RealData& g, real_t bg, RealData& f) override;
    void resetDataStepState() override;

private:
    // Extra scratch for the line search. The unit-step base shares one buffer for
    // m and d (never needed together there); this variant needs m (for J(alpha))
    // and d simultaneously, so it keeps the model in its own buffer and reuses the
    // base's `scratch` for d. hd = H*d, the forward model of the direction.
    RealData model;
    RealData hd;
    ComplexView hdC;

    // Per-deconvolution step-damping state.
    real_t pwr = 1.0f;

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};
