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
        // Application selection
        // Note: Application enum might need special handling or conversion to int
        {"Application", ParameterType::Int, reinterpret_cast<int*>(&setupConfig->app), 0, 1}, // 0=deconvolution, 1=psfgeneration
        
        // File paths - these might need special UI handling for file selection
        {"Image Path", ParameterType::String, &setupConfig->imagePath, 0.0, 0.0},
        {"GPU Type", ParameterType::String, &setupConfig->gpu, 0.0, 0.0},
        
        // Boolean flags
        {"Show Time", ParameterType::Bool, &setupConfig->time, 0.0, 1.0},
        {"Save Layer Separate", ParameterType::Bool, &setupConfig->sep, 0.0, 1.0},
        {"Save PSF", ParameterType::Bool, &setupConfig->savePsf, 0.0, 1.0},
        {"Show Example Layers", ParameterType::Bool, &setupConfig->showExampleLayers, 0.0, 1.0},
        {"Print Info", ParameterType::Bool, &setupConfig->printInfo, 0.0, 1.0},
        {"Save Subimages", ParameterType::Bool, &setupConfig->saveSubimages, 0.0, 1.0},
        
        {"PSF Config Path", ParameterType::String, &setupConfig->psfConfigPath, 0.0, 0.0},
        {"PSF File Path", ParameterType::String, &setupConfig->psfFilePath, 0.0, 0.0},
        {"PSF Directory Path", ParameterType::String, &setupConfig->psfDirPath, 0.0, 0.0},
        // Vector parameters (commented out as they need special handling)
        // {"Layers", ParameterType::IntVector, &setupConfig->layers, 0.0, 0.0},
        // {"Subimages", ParameterType::IntVector, &setupConfig->subimages, 0.0, 0.0},
    };
    parameters.insert(parameters.end(), runtimeParams.begin(), runtimeParams.end());
}
