/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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
    backend = other.backend;
    outputDir = other.outputDir;
    strategyType = other.strategyType;
    labeledImage = other.labeledImage;
    labelPSFMap = other.labelPSFMap;
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
        backend = other.backend;
        outputDir = other.outputDir;
        strategyType = other.strategyType;
        labeledImage = other.labeledImage;
        labelPSFMap = other.labelPSFMap;
        
        // Deep copy the shared_ptr content
        if (other.deconvolutionConfig != nullptr) {
            deconvolutionConfig = std::make_shared<DeconvolutionConfig>(*other.deconvolutionConfig);
        } else {
            deconvolutionConfig.reset();  // Set to nullptr
        }
    }
    return *this;
}

bool SetupConfig::loadFromJSON(const json& jsonData){
    bool success = Config::loadFromJSON(jsonData);

    if (jsonData.contains("Deconvolution")){
        deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
        deconvolutionConfig->loadFromJSON(jsonData["Deconvolution"]);
    }
    else{

        deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
    }
 
    return success;

}


void SetupConfig::registerAllParameters(){
    // Register each parameter as a ConfigParameter struct
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    parameters.push_back({ParameterType::Bool, &sep, "sep", false, "sep", "--sep", "Save layer separate", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &time, "time", false, "time", "--time", "Show duration", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &savePsf, "savePsf", false, "savePsf", "--savePsf", "Save used PSF", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &showExampleLayers, "showExampleLayers", false, "showExampleLayers", "--showExampleLayers", "Show example layers", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &printInfo, "info", false, "info", "--info", "Print info about input image", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &imagePath, "image_path", false, "image_path", "-i,--image_path", "Input image path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &outputDir, "outputDir", true, "outputDir", "--outputDir", "Output directory", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &psfConfigPath, "psf_config_path", true, "psf_config_path", "--psf_config_path", "PSF config path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &psfFilePath, "psf_file_path", true, "psf_file_path", "--psf_file_path", "PSF file path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::DeconvolutionConfig, &deconvolutionConfig, "Deconvolution", true, "DeconConfig", "--deconvConfig", "Deconv Config", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::String, &strategyType, "strategyType", true, "strategyType", "--strategyType", "Deconvolution strategy type", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &labeledImage, "labeledImage", true, "labeledImage", "--labeledImage", "Labeled image path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::String, &labelPSFMap, "labelPSFMap", true, "labelPSFMap", "--labelPSFMap", "Label PSF map path", false, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::Bool, &saveSubimages, "saveSubimages", true, "saveSubimages", "--saveSubimages", "Save subimages separate", false, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::String, &backend, "backend", true, "backend", "--backend", "Backend type", false, false, 0.0, 0.0, nullptr});
    
}