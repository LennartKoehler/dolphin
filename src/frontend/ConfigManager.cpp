#include "frontend/DeconvolutionConfig.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>





void ConfigManager::setCuda(){
#ifdef CUDA_AVAILABLE
    if(gpu == "") {
        gpu = "cuda";
        std::cout << "[INFO] CUDA activated, to deactivated use --gpu none (CPU parallelism is deactivated for deconvtoolcuda)" << std::endl;
    }else if(gpu != "cuda") {
        std::cout << "[WARNING] --gpu set to "<< gpu <<". CUDA is available, but not activated. Use --gpu cuda (CPU parallelism is deactivated for deconvtoolcuda)" << std::endl;
    }
#endif
}



void ConfigManager::handleJSONConfigs(const std::string& configPath) {
    this->config = loadJSONFile(configPath);
    image_path = extractImagePath(this->config);
    processPSFPaths();
    extractAlgorithmParameters();
    extractOptionalParameters();
}

json ConfigManager::loadJSONFile(const std::string& filePath) const {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + filePath);
    }
    
    std::cout << "[STATUS] " << filePath << " successfully read" << std::endl;

    json jsonFile;
    file >> jsonFile;
    return jsonFile;
}

std::string ConfigManager::extractImagePath(const json& config) const {
    return config["image_path"].get<std::string>();
}

void ConfigManager::processPSFPaths() {
    if (!config.contains("psf_path")) {
        return;
    }
    
    if (config["psf_path"].is_string()) {
        std::string psf_path = config["psf_path"].get<std::string>();
        processSinglePSFPath(psf_path);
    } else if (config["psf_path"].is_array()) {
        processPSFPathArray();
    } else {
        throw std::runtime_error("Field 'psf_path' has invalid format.");
    }
}

void ConfigManager::processSinglePSFPath(const std::string& psf_path) {
    
    if (isJSONFile(psf_path)) {
        json configJson = loadJSONFile(psf_path);
        if (configJson.contains("path") && configJson["path"].get<std::string>() != "") {
            psfPaths.push_back(configJson["path"].get<std::string>());
        }
        else{
            psfJSON.push_back(configJson);
        }
    } else {
        // if (!psf_path.empty()) { // LK why do i need this? only add psf_paths if theyre not empty?
        psfPaths.push_back(psf_path);
        // }
    }
}

void ConfigManager::processPSFPathArray() {
    for (const auto& element : config["psf_path"]) {
        if (element.is_string()) {
            std::string elementStr = element.get<std::string>();
            processSinglePSFPath(elementStr);
        }
    }
}

bool ConfigManager::isJSONFile(const std::string& path) {
    return path.substr(path.find_last_of(".") + 1) == "json";
}




void ConfigManager::extractAlgorithmParameters() {
    algorithmName = config["algorithm"].get<std::string>();
    epsilon = config["epsilon"].get<double>();
    iterations = config["iterations"].get<int>();
    lambda = config["lambda"].get<double>();
    psfSafetyBorder = config["psfSafetyBorder"].get<int>();
    subimageSize = config["subimageSize"].get<int>();
    borderType = config["borderType"].get<int>();
}

void ConfigManager::extractOptionalParameters() {
    sep = config["seperate"].get<bool>();
    time = config["time"].get<bool>();
    savePsf = config["savePsf"].get<bool>();
    showExampleLayers = config["showExampleLayers"].get<bool>();
    printInfo = config["info"].get<bool>();
    grid = config["grid"].get<bool>();
    
    if (config.contains("saveSubimages")) {
        saveSubimages = config["saveSubimages"].get<bool>();
    }
    if (config.contains("gpu")) {
        gpu = config["gpu"].get<std::string>();
    }
}

