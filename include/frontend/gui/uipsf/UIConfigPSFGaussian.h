#pragma once
#include "psf/configs/GaussianPSFConfig.h"
#include "UIConfigPSF.h"


class UIConfigPSFGaussian : public UIConfigPSF<GaussianPSFConfig>{
public:
    UIConfigPSFGaussian();
    std::shared_ptr<PSFConfig> getConfig() override;
    void setParameters(std::shared_ptr<const GaussianPSFConfig> config) override;

private:
    void setSpecificParameters(std::shared_ptr<GaussianPSFConfig> config) override;

    std::shared_ptr<GaussianPSFConfig> psfConfig;
};