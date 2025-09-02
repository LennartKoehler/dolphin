#pragma once

#include "UIConfig.h"
#include "frontend/SetupConfig.h"


class UISetupConfig : public UIConfig{
public:
    UISetupConfig();
    void setParameters(std::shared_ptr<SetupConfig> config); // no copy is created, the actual config is edited live
    std::shared_ptr<SetupConfig> getConfig();


private:
    void setSetupConfigParameters(std::shared_ptr<SetupConfig> setupConfig);
    std::shared_ptr<SetupConfig> setupConfig;
};