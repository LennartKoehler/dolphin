#pragma once


#include "psf/PSF.h"
#include "psf/configs/PSFConfig.h"
#include "BasePSFGenerator.h"

class GaussianPSFConfig;

class GaussianPSFGenerator : public BasePSFGenerator {
public:
    GaussianPSFGenerator() = default;
    GaussianPSFGenerator(std::unique_ptr<PSFConfig>&& config) { setConfig(std::move(config)); }

    PSF generatePSF() const override;
    void setConfig(std::unique_ptr<PSFConfig> config) override;
    bool hasConfig() override;

private:
    std::unique_ptr<GaussianPSFConfig> config;
};




