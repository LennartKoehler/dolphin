#pragma once
#include <string>
#include <vector>
#include <memory>
#include "psf/configs/PSFConfig.h"
#include "GUIStyleConfig.h"



class UIPSFConfig{
public:
    UIPSFConfig() = default;
    virtual ~UIPSFConfig();
    virtual void setParameters(std::shared_ptr<PSFConfig> config) = 0;
    virtual std::shared_ptr<PSFConfig> getConfig() = 0;

    void showParameters(std::shared_ptr<GUIStyleConfig> style);

protected:
    virtual void setConfig(std::shared_ptr<PSFConfig> config) = 0;
    void setDefaultParameters(std::shared_ptr<PSFConfig> conifg);
    std::vector<ParameterDescription> params;
};