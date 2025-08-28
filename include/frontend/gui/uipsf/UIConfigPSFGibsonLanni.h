#pragma once
#include "UIConfigPSF.h"
#include "psf/configs/GibsonLanniPSFConfig.h"

class UIConfigPSFGibsonLanni : public UIConfigPSF<GibsonLanniPSFConfig>{
public:
    UIConfigPSFGibsonLanni();
    std::shared_ptr<PSFConfig> getConfig() override;

    void setParameters(const std::shared_ptr<const GibsonLanniPSFConfig> config) override;

private:
    void setSpecificParameters(std::shared_ptr<GibsonLanniPSFConfig> config) override;
};