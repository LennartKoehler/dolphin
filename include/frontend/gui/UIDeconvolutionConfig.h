#pragma once

#include "UIConfig.h"
#include "deconvolution/DeconvolutionConfig.h"


class UIDeconvolutionConfig : public UIConfig{
public:
    UIDeconvolutionConfig();
    void setParameters(std::shared_ptr<const DeconvolutionConfig> config);
    std::shared_ptr<DeconvolutionConfig> getConfig();


private:
    void setDeconvolutionConfigParameters(std::shared_ptr<DeconvolutionConfig> deconvolutionConfig);
    std::shared_ptr<DeconvolutionConfig> deconvolutionConfig;
};