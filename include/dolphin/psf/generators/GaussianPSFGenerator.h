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


#include "dolphin/psf/PSF.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"

class GaussianPSFConfig;

class GaussianPSFGenerator : public BasePSFGenerator {
public:
    GaussianPSFGenerator() = default;
    GaussianPSFGenerator(std::shared_ptr<PSFConfig>&& config) { setConfig(std::move(config)); }

    PSF generatePSF() const override;
    void setConfig(const std::shared_ptr<const PSFConfig> config) override;
    bool hasConfig() override;

private:
    std::shared_ptr<GaussianPSFConfig> config;
};




