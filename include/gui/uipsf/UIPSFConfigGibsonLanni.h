#pragma once
#include "UIPSFConfig.h"
#include "GibsonLanniPSFConfig.h"

class UIPSFConfigGibsonLanni : public UIPSFConfig{
public:
    std::shared_ptr<PSFConfig> getConfig() override;

    void setConfig(std::shared_ptr<PSFConfig> config) override;
    void setParameters(std::shared_ptr<PSFConfig> config) override;

private:
    void setGibsonLanniParameters(std::shared_ptr<GibsonLanniPSFConfig> config);
    std::shared_ptr<GibsonLanniPSFConfig> psfConfig;
};