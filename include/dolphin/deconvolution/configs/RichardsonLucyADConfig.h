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

// Richardson-Lucy with adaptive damping: uses the shared iteration count only.
class RichardsonLucyADConfig : public DeconvolutionConfig {
public:
    RichardsonLucyADConfig();
    RichardsonLucyADConfig(const RichardsonLucyADConfig& other);
    RichardsonLucyADConfig& operator=(const RichardsonLucyADConfig& other);
    RichardsonLucyADConfig(RichardsonLucyADConfig&& other) noexcept;
    // Move-assignment is not offered: it would have to re-register parameters.

    static constexpr const char* kAlgorithmName = DeconvolutionAlgorithms::kRichardsonLucyAdaptiveDamping;

protected:
    void registerAllParameters() override;
};
