#include "frontend/gui/UISetupConfig.h"


UISetupConfig::UISetupConfig(){
    setParameters(std::make_shared<SetupConfig>());
}

void UISetupConfig::setParameters(std::shared_ptr<SetupConfig> config){
    this->setupConfig = config;
    setSetupConfigParameters(config);
}

std::shared_ptr<SetupConfig> UISetupConfig::getConfig(){
    return setupConfig;
}


void UISetupConfig::setSetupConfigParameters(std::shared_ptr<SetupConfig> setupConfig){
    std::vector<ParameterDescription> runtimeParams = {
        // {"Iterations", ParameterType::Int, &setupConfig->iterations, 1, 10000},
        // {"Lambda", ParameterType::Double, &setupConfig->lambda, 0.0, 1.0},
        // {"Epsilon", ParameterType::Double, &setupConfig->epsilon, 1e-12, 1e-3},
        // {"Grid Processing", ParameterType::Bool, &setupConfig->grid, 0.0, 1.0},
        // {"Subimage Size", ParameterType::Int, &setupConfig->subimageSize, 0, 10000},
        // {"PSF Safety Border", ParameterType::Int, &setupConfig->psfSafetyBorder, 0, 1000},
        // {"Border Type", ParameterType::Int, &setupConfig->borderType, 0, 5},
        {"Time", ParameterType::Bool, &setupConfig->time, 0.0, 1.0},
        {"Save Layer Separate", ParameterType::Bool, &setupConfig->sep, 0.0, 1.0},
        {"Save PSF", ParameterType::Bool, &setupConfig->savePsf, 0.0, 1.0},
        {"Show Example Layers", ParameterType::Bool, &setupConfig->showExampleLayers, 0.0, 1.0},
        {"Print Info", ParameterType::Bool, &setupConfig->printInfo, 0.0, 1.0},

        {"Save Subimages", ParameterType::Bool, &setupConfig->saveSubimages, 0.0, 1.0}
    };
    parameters.insert(parameters.end(), runtimeParams.begin(), runtimeParams.end());
}