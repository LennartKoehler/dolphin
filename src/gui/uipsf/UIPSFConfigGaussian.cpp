#include "gui/uipsf/UIPSFConfigGaussian.h"


UIPSFConfigGaussian::UIPSFConfigGaussian(){
    setParameters(std::make_shared<GaussianPSFConfig>());
}

std::shared_ptr<PSFConfig> UIPSFConfigGaussian::getConfig(){
    return psfConfig;
}

void UIPSFConfigGaussian::setParameters(std::shared_ptr<PSFConfig> config){
    setConfig(config);
    setDefaultParameters(config);
    setGaussianParameters(this->psfConfig);
}

void UIPSFConfigGaussian::setConfig(std::shared_ptr<PSFConfig> config){
    auto g = std::dynamic_pointer_cast<GaussianPSFConfig>(config);
    if (!g) {
        throw std::runtime_error("UIPSFConfigGaussian: wrong config type passed!");
    }
    psfConfig = g;
}

void UIPSFConfigGaussian::setGaussianParameters(std::shared_ptr<GaussianPSFConfig> config){
    std::vector<ParameterDescription> temp = {
        {"Sigma X", ParameterType::Double, &config->sigmaX, 0.1, 100.0},
        {"Sigma Y", ParameterType::Double, &config->sigmaY, 0.1, 100.0},
        {"Sigma Z", ParameterType::Double, &config->sigmaZ, 0.1, 100.0},
        {"Quality Factor", ParameterType::Double, &config->qualityFactor, 0.1, 10.0},
        {"Pixel Scaling", ParameterType::Double, &config->pixelScaling, 1e-9, 1e-4},
        {"Nanometer Scale", ParameterType::Double, &config->nanometerScale, 1e-12, 1e-6},
        {"PSF Layers", ParameterType::Vector, &config->psfLayers, 0, 1024},
        {"PSF Cubes", ParameterType::Vector, &config->psfCubes, 0, 1024},
        {"PSF Path", ParameterType::String, &config->psfPath, 0, 0},
        {"PSF Model Name", ParameterType::String, &config->psfModelName, 0, 0}
    };
    params.insert(params.end(), temp.begin(), temp.end());

}

