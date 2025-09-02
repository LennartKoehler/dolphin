#include "frontend/gui/UIDeconvolutionConfig.h"


UIDeconvolutionConfig::UIDeconvolutionConfig(){
    setParameters(std::make_shared<DeconvolutionConfig>());
}

void UIDeconvolutionConfig::setParameters(std::shared_ptr<const DeconvolutionConfig> config){
    auto configCopy = std::make_shared<DeconvolutionConfig>(*config);
    deconvolutionConfig = configCopy;
    setDeconvolutionConfigParameters(configCopy);
}

std::shared_ptr<DeconvolutionConfig> UIDeconvolutionConfig::getConfig(){
    return deconvolutionConfig;
}


void UIDeconvolutionConfig::setDeconvolutionConfigParameters(std::shared_ptr<DeconvolutionConfig> deconvolutionConfig){
    std::vector<ParameterDescription> runtimeParams = {
        {"Iterations", ParameterType::Int, &deconvolutionConfig->iterations, 1, 10000},
        {"Lambda", ParameterType::Double, &deconvolutionConfig->lambda, 0.0, 1.0},
        {"Epsilon", ParameterType::Double, &deconvolutionConfig->epsilon, 1e-12, 1e-3},
        {"Grid Processing", ParameterType::Bool, &deconvolutionConfig->grid, 0.0, 1.0},
        {"Subimage Size", ParameterType::Int, &deconvolutionConfig->subimageSize, 0, 10000},
        {"PSF Safety Border", ParameterType::Int, &deconvolutionConfig->psfSafetyBorder, 0, 1000},
        {"Border Type", ParameterType::Int, &deconvolutionConfig->borderType, 0, 5},

    };
    parameters.insert(parameters.end(), runtimeParams.begin(), runtimeParams.end());
}