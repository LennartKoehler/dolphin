#include "frontend/gui/UIDeconvolutionConfig.h"


UIDeconvolutionConfig::UIDeconvolutionConfig(){
    setParameters(std::make_shared<DeconvolutionConfig>());
}

void UIDeconvolutionConfig::setParameters(std::shared_ptr<const DeconvolutionConfig> config){
    auto configCopy = std::make_shared<DeconvolutionConfig>(*config);
    configManager = configCopy;
    setConfigManagerParameters(configCopy);
}

std::shared_ptr<DeconvolutionConfig> UIDeconvolutionConfig::getConfig(){
    return configManager;
}


void UIDeconvolutionConfig::setConfigManagerParameters(std::shared_ptr<DeconvolutionConfig> configManager){
    std::vector<ParameterDescription> runtimeParams = {
        {"Iterations", ParameterType::Int, &configManager->iterations, 1, 10000},
        {"Lambda", ParameterType::Double, &configManager->lambda, 0.0, 1.0},
        {"Epsilon", ParameterType::Double, &configManager->epsilon, 1e-12, 1e-3},
        {"Time", ParameterType::Bool, &configManager->time, 0.0, 1.0},
        {"Save Layer Separate", ParameterType::Bool, &configManager->sep, 0.0, 1.0},
        {"Save PSF", ParameterType::Bool, &configManager->savePsf, 0.0, 1.0},
        {"Show Example Layers", ParameterType::Bool, &configManager->showExampleLayers, 0.0, 1.0},
        {"Print Info", ParameterType::Bool, &configManager->printInfo, 0.0, 1.0},
        {"Grid Processing", ParameterType::Bool, &configManager->grid, 0.0, 1.0},
        {"Subimage Size", ParameterType::Int, &configManager->subimageSize, 0, 10000},
        {"PSF Safety Border", ParameterType::Int, &configManager->psfSafetyBorder, 0, 1000},
        {"Border Type", ParameterType::Int, &configManager->borderType, 0, 5},
        {"Save Subimages", ParameterType::Bool, &configManager->saveSubimages, 0.0, 1.0}
    };
    parameters.insert(parameters.end(), runtimeParams.begin(), runtimeParams.end());
}