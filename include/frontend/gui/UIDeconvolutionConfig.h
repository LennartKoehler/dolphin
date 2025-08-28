#pragma once

#include "UIConfig.h"
#include "frontend/DeconvolutionConfig.h"


class UIDeconvolutionConfig : public UIConfig{
public:
    UIDeconvolutionConfig();
    void setParameters(std::shared_ptr<const DeconvolutionConfig> config);
    std::shared_ptr<DeconvolutionConfig> getConfig();


private:
    void setConfigManagerParameters(std::shared_ptr<DeconvolutionConfig> configManager);
    std::shared_ptr<DeconvolutionConfig> configManager;
};