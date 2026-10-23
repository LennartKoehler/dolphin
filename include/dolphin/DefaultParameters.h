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
#include "dolphin/Config.h"

// Process-wide defaults loaded once from a JSON file at first use.
// The values are fixed after loading and must be read through the getters.
class DefaultParameters : public Config{
public:

    static const DefaultParameters& getDefaultParameters();

    float lowPaddingThreshold() const { return lowPaddingThreshold_; }
    float mediumPaddingThreshold() const { return mediumPaddingThreshold_; }
    float generationThresholdMultiplier() const { return generationThresholdMultiplier_; }

private:

    DefaultParameters(const DefaultParameters&) = delete;
    DefaultParameters& operator=(const DefaultParameters&) = delete;
    DefaultParameters(){registerAllParameters();}

    std::string getName() const override{
        return "DefaultParameters";
    }
    void registerAllParameters();
    static std::string resolveConfigPath();

    // Built-in fallbacks (used when the JSON file is missing); match the
    // preset padding thresholds used by PSFHandler.
    float lowPaddingThreshold_ = 1e-3f;
    float mediumPaddingThreshold_ = 1e-4f;
    float generationThresholdMultiplier_ = 1e-2f;
};
