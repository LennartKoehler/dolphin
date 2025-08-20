#include "psf/GaussianPSFGenerator.h"
#include <fstream>
#include <iostream>

#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

std::string GaussianPSFConfig::getName(){
    return this->psfModelName;
}

bool GaussianPSFConfig::loadFromJSON(const json& jsonData) {

    try {

        loadFromJSONBase(jsonData);
        if (jsonData.contains("qualityFactor")){
            this->qualityFactor = jsonData["qualityFactor"].get<double>();
        }


        // // LK TODO do these need to be here?
        // if (jsonData.contains("layers")) {
        //     this->psfLayers = jsonData["layers"].get<std::vector<int>>();

        // } else {
        //     //throw std::runtime_error("Missing required parameter: secondpsflayers");
        // }
        // if (jsonData.contains("subimages")) {
        //     this->psfCubes = jsonData["subimages"].get<std::vector<int>>();

        // } else {
        //     //throw std::runtime_error("Missing required parameter: secondpsfcubes");
        // }




        if (!jsonData.contains("sigmaX")){
            if (!resAxial_nm){
                throw std::runtime_error("[ERROR] Missing required parameter: sigmax or resAxial");
            }
            this->sigmaX = convertResolution(resAxial_nm);
        }
        else{this->sigmaX = convertSigma(jsonData.at("sigmaX").get<double>());}


        if (!jsonData.contains("sigmaY")){
            if (!resAxial_nm){
                throw std::runtime_error("[ERROR] Missing required parameter: sigmay or resolutiony");
            }
            this->sigmaY = convertResolution(resAxial_nm);
        }
        else{this->sigmaY = convertSigma(jsonData.at("sigmaY").get<double>());}

        if (!jsonData.contains("sigmaZ")){
            if (!resLateral_nm){
                throw std::runtime_error("[ERROR] Missing required parameter: sigmaz or resolutionz");
            }
            this->sigmaZ = convertResolution(resLateral_nm);
        }
        else{this->sigmaZ = convertSigma(jsonData.at("sigmaZ").get<double>());}



    } catch (const json::exception &e) {
        std::cerr << "[ERROR] Invalid PSF JSON structure: " << e.what() << std::endl;
        return false;
    }
    return true;
}



void GaussianPSFConfig::printValues() {
    std::cout << "[INFO] Gaussian PSF parameters loaded from JSON file:" << std::endl;
    std::cout << "  sigmax: " << this->sigmaX << ", sigmay: " << this->sigmaY
              << ", sigmaz: " << this->sigmaZ << std::endl;
    std::cout << "  psfx: " << this->sizeX << ", psfy: " << this->sizeY
              << ", psfz: " << this->sizeZ << std::endl;


    // Check values
    std::cout << "[STATUS] secondpsflayers: ";
    for (const int& layer : this->psfLayers) {
        std::cout << layer << " ";
    }
    std::cout << std::endl;

    std::cout << "[STATUS] secondpsfcubes: ";
    for (const int& cube : this->psfCubes) {
        std::cout << cube << " ";
    }
    std::cout << std::endl;
}

double GaussianPSFConfig::convertResolution(double resolution_nm){
    return convertSigma(resolution_nm * nanometerScale / pixelScaling);
}

double GaussianPSFConfig::convertSigma(double sigma){
    return sigma * qualityFactor;
}

