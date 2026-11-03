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

#include "dolphin/deconvolution/algorithms/PoissonMLEUnitStepDeconvolutionAlgorithm.h"
#include "dolphin/backend/BackendFactory.h"

void PoissonMLEUnitStepDeconvolutionAlgorithm::applyDataStep(
    const IComputeBackend& compute, const ComplexData& /*H*/, const RealData& /*g*/,
    const RealData& /*model*/, RealData& direction, RealData& f) {
    // Plain RL unit step: f <- f + d, clipped at zero. No hd = H*d forward model
    // and no line search, so this iteration is much cheaper than the base class's.
    compute.realLinearCombination(f, 1.0f, direction, 1.0f, f);
    compute.realClampMin(f, 0.0f, f);
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEUnitStepDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEUnitStepDeconvolutionAlgorithm>();
    copy->iterations = iterations;
    copy->snr = snr;
    copy->lambdaOverride = lambdaOverride;
    copy->backgroundOverride = backgroundOverride;
    copy->usePrefilter = usePrefilter;
    copy->prefilterScale = prefilterScale;
    copy->useSieve = useSieve;
    copy->usePrior = usePrior;
    copy->sievePasses = sievePasses;
    copy->sieveScale = sieveScale;
    copy->initialized = false;
    return copy;
}