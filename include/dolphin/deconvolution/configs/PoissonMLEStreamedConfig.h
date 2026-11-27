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

#include "dolphin/deconvolution/configs/PoissonMLEUnitStepConfig.h"

// Config for PoissonMLEStreamed: identical parameters to PoissonMLEUnitStepConfig
// (same math, out-of-core data step). Exists as its own class so the config's
// algorithmName matches the selectable algorithm name.
class PoissonMLEStreamedConfig : public PoissonMLEUnitStepConfig {
public:
    PoissonMLEStreamedConfig();
    PoissonMLEStreamedConfig(const PoissonMLEStreamedConfig& other);
    PoissonMLEStreamedConfig& operator=(const PoissonMLEStreamedConfig& other);

    static constexpr const char* kAlgorithmName = DeconvolutionAlgorithms::kPoissonMLEStreamed;
};
