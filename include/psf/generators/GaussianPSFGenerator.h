#pragma once


#include "psf/PSF.h"
#include "psf/configs/PSFConfig.h"
#include "BasePSFGenerator.h"

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




