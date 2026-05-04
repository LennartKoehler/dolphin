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
#include "dolphin/HelperClasses.h"
#include "dolphin/Config.h"


enum PaddingFillType {
    ZERO,
    MIRROR,
    LINEAR,
    QUADRATIC,
    SINUSOID,
    GAUSSIAN,
};

enum PaddingStrategyType {
    NONE,
    PARENT,
    FULL_PSF,
    MANUAL,
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
    float epsilon = 1e-6;
    float lambda = 0.001;
    PaddingFillType paddingFillType = PaddingFillType::ZERO;
    PaddingStrategyType paddingStrategyType = PaddingStrategyType::PARENT;
    float paddingRelativeMax = 0.001;
    int featheringRadius = 0;
    std::array<int, 3> cubeSize{}; // currently unused
    std::array<int, 3> cubePadding{-1, -1, -1}; // this padding is later doubled

    // virtual bool loadFromJSON(const json& jsonData) override;
    // virtual json writeToJSON() const override;
private:
    virtual void registerAllParameters();
};
