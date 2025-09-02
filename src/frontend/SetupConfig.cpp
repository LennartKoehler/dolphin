#include "frontend/SetupConfig.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include "DeconvolutionConfig.h"
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
    psfConfigPath = readParameter<std::string>(config, "psf_config_path");
    psfFilePath = readParameter<std::string>(config, "psf_file_path");
    psfDirPath = readParameter<std::string>(config, "psf_dir_path");
    
    // Arrays
    layers = readParameter<std::vector<int>>(config, "layers");
    subimages = readParameter<std::vector<int>>(config, "subimages");

    // Optional fields using readParameterOptional
    readParameterOptional<bool>(config, "saveSubimages", saveSubimages);
    readParameterOptional<std::string>(config, "gpu", gpu);
    
    // Optional fields from Deconvolution section
    if (config.contains("Deconvolution")) {
        deconvolutionConfig->loadFromJSON(config.at("Deconvolution"));
    }
    else {
        throw std::runtime_error("[ERROR] Missing required parameter: Deconvolution");
    }
    if (config.contains("PSF")) {
        psfConfig = PSFConfig::createFromJSON(config);
    }

    
    return true;
}


// SetupConfig::SetupConfig(const SetupConfig& other) 
//     : Config(other)  // Call base class copy constructor
//     , app(other.app)
//     , imagePath(other.imagePath)
//     , psfConfigPath(other.psfConfigPath)
//     , psfFilePath(other.psfFilePath)
//     , psfDirPath(other.psfDirPath)
//     , time(other.time)
//     , sep(other.sep)
//     , savePsf(other.savePsf)
//     , showExampleLayers(other.showExampleLayers)
//     , printInfo(other.printInfo)
//     , saveSubimages(other.saveSubimages)
//     , gpu(other.gpu)
//     , layers(other.layers)
//     , subimages(other.subimages)
//     , deconvolutionConfig(other.deconvolutionConfig ? 
//                          std::make_unique<DeconvolutionConfig>(*other.deconvolutionConfig) : 
//                          nullptr)
//     , psfConfig(other.psfConfig ? 
//                other.psfConfig->clone() :  // Assumes PSFConfig has clone method
//                nullptr)
// {
// }

// // Copy assignment operator
// SetupConfig& SetupConfig::operator=(const SetupConfig& other) {
//     if (this != &other) {  // Self-assignment check
//         Config::operator=(other);  // Call base class assignment
        
//         app = other.app;
//         imagePath = other.imagePath;
//         psfConfigPath = other.psfConfigPath;
//         psfFilePath = other.psfFilePath;
//         psfDirPath = other.psfDirPath;
//         time = other.time;
//         sep = other.sep;
//         savePsf = other.savePsf;
//         showExampleLayers = other.showExampleLayers;
//         printInfo = other.printInfo;
//         saveSubimages = other.saveSubimages;
//         gpu = other.gpu;
//         layers = other.layers;
//         subimages = other.subimages;
        
//         // Deep copy unique_ptrs
//         deconvolutionConfig = other.deconvolutionConfig ? 
//                              std::make_unique<DeconvolutionConfig>(*other.deconvolutionConfig) : 
//                              nullptr;
//         psfConfig = other.psfConfig ? 
//                    other.psfConfig->clone() : 
//                    nullptr;
//     }
//     return *this;
// }