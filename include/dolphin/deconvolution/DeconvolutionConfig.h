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

#include <string>
#include "dolphin_image/HelperClasses.h"
#include "dolphin/Config.h"
#include "dolphin_image/Types/PaddingFillType.h"

// PaddingFillType is now defined in dolphin_image/Types/PaddingFillType.h
// It is still available here for backward compatibility.

// Preset PSF padding thresholds, expressed as a fraction of the PSF max value.
// Setting cube_padding (all values >= 0) or padding_relative_max (> 0)
// overrides the preset; see PSFHandler::getPadding for resolution order.
enum PaddingStrategyType {
    DEFAULT, // to track if it was set
    NONE,
    LOW,
    MEDIUM,
    MAX
};

class DeconvolutionConfig : public Config{
public:
    DeconvolutionConfig();
    DeconvolutionConfig(const DeconvolutionConfig& other);
    DeconvolutionConfig& operator=(const DeconvolutionConfig& other);
    DeconvolutionConfig(DeconvolutionConfig&& other) noexcept;
    DeconvolutionConfig& operator=(DeconvolutionConfig&& other) noexcept;
    static DeconvolutionConfig createFromJSONFile(const std::string& path);
    std::string getName() const override { return std::string("DeconvolutionConfig"); };

    // Use the struct for parameters
    std::string algorithmName = "RichardsonLucy";
    int iterations = 10;
    float minDenominator = 1e-6f;
    float signalToNoise = 10.0f;
    PaddingFillType paddingFillType = PaddingFillType::ZERO;
    PaddingStrategyType paddingStrategyType = PaddingStrategyType::DEFAULT;
    float paddingRelativeMax = 0.0f; // > 0: fraction of PSF max to pad to; 0 = use preset
    int featheringRadius = 0;
    std::array<size_t, 3> cubeSize{0, 0, 0}; // currently unused
    std::optional<std::array<size_t, 3>> cubePadding = std::nullopt;
    bool sliceAllDimensions = false;

    // --- PoissonMLE (Poisson / I-divergence likelihood) specific parameters ---
    // Shrinkage strength lambda for the prior step. 0 => derive from signal_to_noise.
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

private:
    virtual void registerAllParameters();
};
