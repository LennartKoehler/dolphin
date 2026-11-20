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

#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"

// PoissonMLE with the plain Richardson-Lucy unit step instead of the Newton line
// search. The parameters are identical to PoissonMLEConfig; only the data step
// of the algorithm differs.
class PoissonMLEUnitStepConfig : public PoissonMLEConfig {
public:
    PoissonMLEUnitStepConfig();
    PoissonMLEUnitStepConfig(const PoissonMLEUnitStepConfig& other);
    PoissonMLEUnitStepConfig& operator=(const PoissonMLEUnitStepConfig& other);
    PoissonMLEUnitStepConfig(PoissonMLEConfig&& other) noexcept;

    static constexpr const char* kAlgorithmName = DeconvolutionAlgorithms::kPoissonMLE;
};
