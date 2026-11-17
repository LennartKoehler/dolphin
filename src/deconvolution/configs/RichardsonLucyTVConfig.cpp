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

#include "dolphin/deconvolution/configs/RichardsonLucyTVConfig.h"

RichardsonLucyTVConfig::RichardsonLucyTVConfig() {
    algorithmName = kAlgorithmName;
    registerAllParameters();
}

RichardsonLucyTVConfig::RichardsonLucyTVConfig(const RichardsonLucyTVConfig& other)
    : DeconvolutionConfig(other) {
}

RichardsonLucyTVConfig& RichardsonLucyTVConfig::operator=(const RichardsonLucyTVConfig& other) {
    if (this != &other) {
        assignBase(other);
    }
    return *this;
}

RichardsonLucyTVConfig::RichardsonLucyTVConfig(RichardsonLucyTVConfig&& other) noexcept
    : DeconvolutionConfig(std::move(other)) {
}

void RichardsonLucyTVConfig::registerAllParameters() {
    DeconvolutionConfig::registerAllParameters();
}
