#include "frontend/gui/uipsf/UIConfigPSFGaussian.h"


UIConfigPSFGaussian::UIConfigPSFGaussian(){
    setParameters(std::make_shared<GaussianPSFConfig>());
}

std::shared_ptr<PSFConfig> UIConfigPSFGaussian::getConfig(){
    return std::make_shared<GaussianPSFConfig>(*psfConfig);
}

void UIConfigPSFGaussian::setParameters(const std::shared_ptr<const GaussianPSFConfig> config){

    auto configCopy = std::make_shared<GaussianPSFConfig>(*config);
    psfConfig = configCopy;

    setDefaultParameters(configCopy);
    setSpecificParameters(configCopy);
}



void UIConfigPSFGaussian::setSpecificParameters(std::shared_ptr<GaussianPSFConfig> config){
    std::vector<ParameterDescription> temp = {
        {"Sigma X", ParameterType::Double, &config->sigmaX, 0.1, 100.0},
        {"Sigma Y", ParameterType::Double, &config->sigmaY, 0.1, 100.0},
        {"Sigma Z", ParameterType::Double, &config->sigmaZ, 0.1, 100.0},
        {"Quality Factor", ParameterType::Double, &config->qualityFactor, 0.1, 10.0},
        {"Pixel Scaling", ParameterType::Double, &config->pixelScaling, 1e-9, 1e-4},
    };
    parameters.insert(parameters.end(), temp.begin(), temp.end());

}

