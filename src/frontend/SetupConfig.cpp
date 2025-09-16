#include "frontend/SetupConfig.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include "deconvolution/DeconvolutionConfig.h"
#include "psf/configs/PSFConfig.h"
#include "psf/PSFGeneratorFactory.h"

SetupConfig::SetupConfig() {
    registerAllParameters();
}




SetupConfig SetupConfig::createFromJSONFile(const std::string& filePath) {
    json jsonData = loadJSONFile(filePath);
    
    SetupConfig config;
    if (!config.loadFromJSON(jsonData)) {
        throw std::runtime_error("Failed to parse config file: " + filePath);
    }
    
    return config;
}

SetupConfig::SetupConfig(const SetupConfig& other) 
    : Config(other)  // Copy base class
{
    // First, register all parameters to set up the infrastructure
    registerAllParameters();
    
    // Then copy all the values
    imagePath = other.imagePath;
    psfConfigPath = other.psfConfigPath;
    psfFilePath = other.psfFilePath;
    psfDirPath = other.psfDirPath;
    time = other.time;
    sep = other.sep;
    savePsf = other.savePsf;
    showExampleLayers = other.showExampleLayers;
    printInfo = other.printInfo;
    saveSubimages = other.saveSubimages;
    gpu = other.gpu;

    // Deep copy the shared_ptr content
    if (other.deconvolutionConfig != nullptr) {
        deconvolutionConfig = std::make_shared<DeconvolutionConfig>(*other.deconvolutionConfig);
    }
    // If other.deconvolutionConfig is nullptr, our deconvolutionConfig will remain nullptr (from registerAllParameters)
}


// Copy assignment operator (recommended to implement if you have copy constructor)
SetupConfig& SetupConfig::operator=(const SetupConfig& other) {
    if (this != &other) {  // Self-assignment check
        Config::operator=(other);  // Copy base class
        
        imagePath = other.imagePath;
        psfConfigPath = other.psfConfigPath;
        psfFilePath = other.psfFilePath;
        psfDirPath = other.psfDirPath;
        time = other.time;
        sep = other.sep;
        savePsf = other.savePsf;
        showExampleLayers = other.showExampleLayers;
        printInfo = other.printInfo;
        saveSubimages = other.saveSubimages;
        gpu = other.gpu;
        
        // Deep copy the shared_ptr content
        if (other.deconvolutionConfig != nullptr) {
            deconvolutionConfig = std::make_shared<DeconvolutionConfig>(*other.deconvolutionConfig);
        } else {
            deconvolutionConfig.reset();  // Set to nullptr
        }
    }
    return *this;
}


void SetupConfig::registerDeconvolution(){
    if (deconvolutionConfig == nullptr){
        deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
    }
    ReadWriteHelper param;
    std::string jsonTag = "Deconvolution";
    param.jsonTag = jsonTag;
    param.reader = [this, jsonTag](const json& jsonData) {
        if (jsonData.contains(jsonTag)) {
            this->deconvolutionConfig->loadFromJSON(jsonData.at(jsonTag));
        }
        else {
            this->deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
            std::cout << "[INFO] No deconvolution parameters found, running with default parameters" << std::endl;
        }
    };
    
    // Writer lambda
    param.writer = [this, jsonTag](ordered_json& jsonData) {
        jsonData[jsonTag] = this->deconvolutionConfig->writeToJSON();
    };
    
    parameters.push_back(std::move(param));
}

void SetupConfig::registerAllParameters(){
    bool optional = true;
    
    
    // Register all parameters
    // Required fields
    registerParameter("seperate", sep, !optional);
    registerParameter("time", time, !optional);
    registerParameter("savePsf", savePsf, !optional);
    registerParameter("showExampleLayers", showExampleLayers, !optional);
    registerParameter("info", printInfo, !optional);
    registerParameter("image_path", imagePath, !optional);
    
    // Optional path configurations
    registerParameter("psf_config_path", psfConfigPath, optional);
    registerParameter("psf_file_path", psfFilePath, optional);
    registerParameter("psf_dir_path", psfDirPath, optional);
    
    // Optional fields
    registerParameter("saveSubimages", saveSubimages, optional);
    registerParameter("gpu", gpu, optional);
    
    // Arrays (if needed)
    // registerParameter("layers", layers, optional);
    // registerParameter("subimages", subimages, optional);
    registerDeconvolution();

}