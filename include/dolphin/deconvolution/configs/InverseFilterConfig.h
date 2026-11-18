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

// Inverse filtering: regularizes the division with the shared minDenominator.
class InverseFilterConfig : public DeconvolutionConfig {
public:
    InverseFilterConfig();
    InverseFilterConfig(const InverseFilterConfig& other);
    InverseFilterConfig& operator=(const InverseFilterConfig& other);
    InverseFilterConfig(InverseFilterConfig&& other) noexcept;
    // Move-assignment is not offered: it would have to re-register parameters.

    static constexpr const char* kAlgorithmName = DeconvolutionAlgorithms::kInverseFilter;

protected:
    void registerAllParameters() override;
};
