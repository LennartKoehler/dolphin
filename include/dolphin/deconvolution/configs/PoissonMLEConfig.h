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

#include "dolphin/deconvolution/DeconvolutionConfig.h"

// Poisson / I-divergence (photon-counting) likelihood deconvolution. These are
// the parameters that only make sense for the PoissonMLE family:
//   data step on J(f) = sum [ g ln(g/m) - g + m ],  m = Hf + bg,  f >= bg,
//   followed by an optional prefilter, an optional lambda-shrinkage prior and an
//   optional annealed Gaussian "sieve". See PoissonMLEDeconvolutionAlgorithm.h
//   for the exact iteration.
//
// PoissonMLEUnitStepConfig derives from this class: it shares every parameter
// (only the data step differs), so the UnitStep variant keeps working unchanged
// and may add its own parameters later without touching this class.
class PoissonMLEConfig : public DeconvolutionConfig {
public:
    PoissonMLEConfig();
    PoissonMLEConfig(const PoissonMLEConfig& other);
    PoissonMLEConfig& operator=(const PoissonMLEConfig& other);
    PoissonMLEConfig(PoissonMLEConfig&& other) noexcept;
    // Move-assignment is not offered: it would have to re-register parameters.

    static constexpr const char* kAlgorithmName = DeconvolutionAlgorithms::kPoissonMLE;

    // Shrinkage strength lambda for the prior step. 0 => derive from signalToNoise.
    float poissonLambda = 0.0f;
    // Background level bg. <0 => estimate from the data (min of the prefiltered
    // estimate when the prefilter is on, else the mean of the dimmest decile).
    float poissonBackground = -1.0f;
    // Enable the prefilter: a single mild anisotropic Gaussian low-pass of the raw
    // data g, used as the MLE's initial estimate f0.
    bool poissonPrefilter = true;
    // Prefilter width scale: per-axis sigma = poissonPrefilterScale * PSF sigma.
    // 0.3 reproduces the reference ("sigma ~= 0.3 * PSF width"); 0 disables the blur.
    float poissonPrefilterScale = 0.3f;
    // Enable the annealed anisotropic Gaussian roughness ("sieve") step.
    bool poissonSieve = true;
    // Enable the lambda-shrinkage + flux-conservation prior step.
    bool poissonPrior = true;
    // Number of sieve smoothing passes per iteration.
    int poissonSievePasses = 2;
    // Sieve peak width as a fraction of the PSF width (per-axis) at SNR=1, annealed
    // to 0 over the run. It is coupled to SNR like the prior: effective scale =
    // poissonSieveScale / snr, so the sieve is as wide as this fraction at snr=1 and
    // fades to the identity (pure MLE) as snr -> inf. Lower base = sharper / more
    // contrast; 0 disables the sieve.
    float poissonSieveScale = 0.1f;

protected:
    void registerAllParameters() override;
};