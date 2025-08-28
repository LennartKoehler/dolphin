#pragma once
#include <string>
#include <vector>
#include <memory>
#include "psf/configs/PSFConfig.h"
#include "frontend/gui/GUIStyleConfig.h"

#include "frontend/gui/UIConfig.h"


template<typename psfT>
class UIConfigPSF : public UIConfig{
    static_assert(std::is_base_of<PSFConfig, psfT>::value,
                "psfT must derive from PSFConfig");
public:
    UIConfigPSF() = default;
    virtual ~UIConfigPSF(){}
    virtual void setParameters(const std::shared_ptr<const psfT> config) = 0;
    virtual std::shared_ptr<PSFConfig> getConfig() = 0;


protected:
    void setDefaultParameters(std::shared_ptr<PSFConfig> config){
        std::vector<ParameterDescription> temp = {
            {"Size X", ParameterType::Int, &config->sizeX, 1, 1024},
            {"Size Y", ParameterType::Int, &config->sizeY, 1, 1024},
            {"Size Z", ParameterType::Int, &config->sizeZ, 1, 512},
            {"NA", ParameterType::Double, &config->NA, 0.1, 2.0},
            {"Lateral resolution (nm)", ParameterType::Double, &config->resLateral_nm, 10.0, 500.0},
            {"Axial resolution (nm)", ParameterType::Double, &config->resAxial_nm, 50.0, 2000.0}
        };
        parameters.insert(parameters.end(), temp.begin(), temp.end());
    }
    virtual void setSpecificParameters(std::shared_ptr<psfT> config) = 0;
    std::shared_ptr<psfT> psfConfig;
};