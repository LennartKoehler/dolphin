#include "frontend/gui/uipsf/UIConfigPSFGaussian.h"


UIConfigPSFGaussian::UIConfigPSFGaussian(){
    setParameters(std::make_shared<GaussianPSFConfig>()); // unnecessary copy
}

std::shared_ptr<PSFConfig> UIConfigPSFGaussian::getConfig(){
    return psfConfig;
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
        {"Nanometer Scale", ParameterType::Double, &config->nanometerScale, 1e-12, 1e-6},
        {"PSF Layers", ParameterType::VectorInt, &config->psfLayers, 0, 1024},
        {"PSF Cubes", ParameterType::VectorInt, &config->psfCubes, 0, 1024},
        {"PSF Path", ParameterType::String, &config->psfPath, 0, 0},
        {"PSF Model Name", ParameterType::String, &config->psfModelName, 0, 0}
    };
    parameters.insert(parameters.end(), temp.begin(), temp.end());

}

