#include "frontend/SetupConfig.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include "deconvolution/DeconvolutionConfig.h"
#include "psf/configs/PSFConfig.h"
#include "psf/PSFGeneratorFactory.h"

bool SetupConfig::loadFromJSON(const json& config){
    // Required fields using readParameter (will throw if missing)
    sep = readParameter<bool>(config, "seperate");
    time = readParameter<bool>(config, "time");
    savePsf = readParameter<bool>(config, "savePsf");
    showExampleLayers = readParameter<bool>(config, "showExampleLayers");
    printInfo = readParameter<bool>(config, "info");
    imagePath = readParameter<std::string>(config, "image_path");
    
    // Path configurations
    readParameterOptional<std::string>(config, "psf_config_path", psfConfigPath);
    readParameterOptional<std::string>(config, "psf_file_path", psfFilePath);
    readParameterOptional<std::string>(config, "psf_dir_path", psfDirPath);
    
    // Arrays
    // layers = readParameter<std::vector<int>>(config, "layers");
    // subimages = readParameter<std::vector<int>>(config, "subimages");

    // Optional fields using readParameterOptional
    readParameterOptional<bool>(config, "saveSubimages", saveSubimages);
    readParameterOptional<std::string>(config, "gpu", gpu);
    
    // Optional fields from Deconvolution section
    if (config.contains("Deconvolution")) {
        deconvolutionConfig->loadFromJSON(config.at("Deconvolution"));
    }
    else {
        deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
        std::cout << "[INFO] No deconvolution parameters found, running with default parameters" << std::endl;
    }
    
    return true;
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
    : Config(other),  // Copy base class
        imagePath(other.imagePath),
        psfConfigPath(other.psfConfigPath),
        psfFilePath(other.psfFilePath),
        psfDirPath(other.psfDirPath),
        time(other.time),
        sep(other.sep),
        savePsf(other.savePsf),
        showExampleLayers(other.showExampleLayers),
        printInfo(other.printInfo),
        saveSubimages(other.saveSubimages),
        gpu(other.gpu)
{
    // Deep copy the shared_ptr content
    if (other.deconvolutionConfig) {
        deconvolutionConfig = std::make_shared<DeconvolutionConfig>(*other.deconvolutionConfig);
    }
    // If other.deconvolutionConfig is nullptr, our deconvolutionConfig will also be nullptr (default)
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
        if (other.deconvolutionConfig) {
            deconvolutionConfig = std::make_shared<DeconvolutionConfig>(*other.deconvolutionConfig);
        } else {
            deconvolutionConfig.reset();  // Set to nullptr
        }
    }
    return *this;
}