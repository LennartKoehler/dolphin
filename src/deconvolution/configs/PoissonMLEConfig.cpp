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

#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"

PoissonMLEConfig::PoissonMLEConfig() {
    algorithmName = kAlgorithmName;
    registerAllParameters();
}

PoissonMLEConfig::PoissonMLEConfig(const PoissonMLEConfig& other)
    : DeconvolutionConfig(other),
      poissonLambda(other.poissonLambda),
      poissonBackground(other.poissonBackground),
      poissonPrefilter(other.poissonPrefilter),
      poissonPrefilterScale(other.poissonPrefilterScale),
      poissonSieve(other.poissonSieve),
      poissonPrior(other.poissonPrior),
      poissonSievePasses(other.poissonSievePasses),
      poissonSieveScale(other.poissonSieveScale) {
    // DeconvolutionConfig(other) registered only the shared parameters.
    registerAllParameters();
}

PoissonMLEConfig& PoissonMLEConfig::operator=(const PoissonMLEConfig& other) {
    if (this != &other) {
        assignBase(other); // clears the registered parameters
        poissonLambda = other.poissonLambda;
        poissonBackground = other.poissonBackground;
        poissonPrefilter = other.poissonPrefilter;
        poissonPrefilterScale = other.poissonPrefilterScale;
        poissonSieve = other.poissonSieve;
        poissonPrior = other.poissonPrior;
        poissonSievePasses = other.poissonSievePasses;
        poissonSieveScale = other.poissonSieveScale;
        registerAllParameters();
    }
    return *this;
}

PoissonMLEConfig::PoissonMLEConfig(PoissonMLEConfig&& other) noexcept
    : DeconvolutionConfig(std::move(other)),
      poissonLambda(other.poissonLambda),
      poissonBackground(other.poissonBackground),
      poissonPrefilter(other.poissonPrefilter),
      poissonPrefilterScale(other.poissonPrefilterScale),
      poissonSieve(other.poissonSieve),
      poissonPrior(other.poissonPrior),
      poissonSievePasses(other.poissonSievePasses),
      poissonSieveScale(other.poissonSieveScale) {
    registerAllParameters();
}

void PoissonMLEConfig::registerAllParameters() {
    DeconvolutionConfig::registerAllParameters();

    parameters.push_back({ParameterType::Float, &poissonLambda, "Poisson Lambda", true, "poisson_lambda", "--poisson_lambda",
        "Shrinkage strength for PoissonMLE prior step (0 = derive from SNR)", false, true, 0.0, 1e4, nullptr});
    parameters.push_back({ParameterType::Float, &poissonBackground, "Poisson Background", true, "poisson_background", "--poisson_background",
        "Background level for PoissonMLE (<0 = estimate from data)", false, true, -1.0, 1e8, nullptr});
    parameters.push_back({ParameterType::Bool, &poissonPrefilter, "Poisson Prefilter", true, "poisson_prefilter", "--poisson_prefilter",
        "Enable prefilter: mild anisotropic Gaussian low-pass of the data used as the initial estimate f0 (auto background = min of f0)", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Float, &poissonPrefilterScale, "Poisson Prefilter Scale", true, "poisson_prefilter_scale", "--poisson_prefilter_scale",
        "Prefilter width: per-axis sigma = scale * PSF sigma (0.3 reproduces the reference; 0 disables the blur)", false, true, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Bool, &poissonSieve, "Poisson Sieve", true, "poisson_sieve", "--poisson_sieve",
        "Enable annealed anisotropic Gaussian smoothing (sieve) in PoissonMLE", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &poissonPrior, "Poisson Prior", true, "poisson_prior", "--poisson_prior",
        "Enable lambda-shrinkage + flux conservation prior step in PoissonMLE", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Int, &poissonSievePasses, "Poisson Sieve Passes", true, "poisson_sieve_passes", "--poisson_sieve_passes",
        "Number of sieve smoothing passes per iteration in PoissonMLE", false, true, 0.0, 64.0, nullptr});
    parameters.push_back({ParameterType::Float, &poissonSieveScale, "Poisson Sieve Scale", true, "poisson_sieve_scale", "--poisson_sieve_scale",
        "Sieve peak width as a fraction of the PSF at SNR=1 (effective = scale/SNR); 0.1 default, lower = sharper / more contrast, 0 disables", false, true, 0.0, 100.0, nullptr});
}