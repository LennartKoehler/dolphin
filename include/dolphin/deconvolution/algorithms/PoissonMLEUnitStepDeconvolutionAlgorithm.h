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

// PoissonMLE with a plain Richardson-Lucy unit step (step = 1) instead of the
// exact Newton line search. The data step becomes f <- f + d (clipped at zero),
// where d = -f*(1 - H^T(g/m)); this is the multiplicative RL update written
// additively, and it provably does not increase the Poisson (I-divergence)
// objective. Because no step length is searched for, the hd = H*d forward model
// and the ~30 full-volume J(alpha) evaluations per iteration are skipped entirely,
// making each iteration markedly cheaper than PoissonMLE (which trades that cost
// for potentially fewer iterations). Prior and sieve steps are identical.
class PoissonMLEUnitStepDeconvolutionAlgorithm : public PoissonMLEDeconvolutionAlgorithm {
public:
    PoissonMLEUnitStepDeconvolutionAlgorithm() = default;
    ~PoissonMLEUnitStepDeconvolutionAlgorithm() = default;

protected:
    void applyDataStep(const IComputeBackend& compute, const ComplexData& H,
                       const RealData& g, const RealData& model,
                       RealData& direction, RealData& f) override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};