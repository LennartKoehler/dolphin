#pragma once
#include "GaussianPSFConfig.h"
#include "UIPSFConfig.h"


class UIPSFConfigGaussian : public UIPSFConfig{
public:
    std::shared_ptr<PSFConfig> getConfig() override;
    void setConfig(std::shared_ptr<PSFConfig>) override;

    void setParameters(std::shared_ptr<PSFConfig> config) override;

private:
    std::shared_ptr<GaussianPSFConfig> psfConfig;
    void setGaussianParameters(std::shared_ptr<GaussianPSFConfig> config);
};