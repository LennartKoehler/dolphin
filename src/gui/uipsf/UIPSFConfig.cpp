#include "gui/uipsf/UIPSFConfig.h"

void UIPSFConfig::showParameters(std::shared_ptr<GUIStyleConfig> style){
    for (const auto parameter : params){
        style->drawParameter(parameter);
    }
}


void UIPSFConfig::setDefaultParameters(std::shared_ptr<PSFConfig> config){
    std::vector<ParameterDescription> temp = {
        {"Size X", ParameterType::Int, &config->sizeX, 1, 1024},
        {"Size Y", ParameterType::Int, &config->sizeY, 1, 1024},
        {"Size Z", ParameterType::Int, &config->sizeZ, 1, 512},
        {"NA", ParameterType::Double, &config->NA, 0.1, 2.0},
        {"Lateral resolution (nm)", ParameterType::Double, &config->resLateral_nm, 10.0, 500.0},
        {"Axial resolution (nm)", ParameterType::Double, &config->resAxial_nm, 50.0, 2000.0}
    };
    params.insert(params.end(), temp.begin(), temp.end());
}