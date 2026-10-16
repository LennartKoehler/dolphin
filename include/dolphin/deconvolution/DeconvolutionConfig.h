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
    float min_denominator = 1e-6f;
    float signal_to_noise = 0.1f;
    PaddingFillType paddingFillType = PaddingFillType::ZERO;
    PaddingStrategyType paddingStrategyType = PaddingStrategyType::DEFAULT;
    float paddingRelativeMax = 0.0f; // > 0: fraction of PSF max to pad to; 0 = use preset
    int featheringRadius = 0;
    // std::array<int, 3> cubeSize{}; // currently unused
    std::array<int, 3> cubePadding{-1, -1, -1}; // this padding is later doubled
    bool sliceAllDimensions = false;

private:
    virtual void registerAllParameters();
};
