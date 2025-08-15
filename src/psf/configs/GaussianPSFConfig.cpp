#include "GaussianPSFGenerator.h"
#include <fstream>
#include <iostream>

#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

std::string GaussianPSFConfig::getName(){
    return this->psfModelName;
}

bool GaussianPSFConfig::loadFromJSON(const json& jsonData) {

    try {

        if (jsonData.contains("qualityFactor")){
            this->qualityFactor = jsonData["qualityFactor"].get<double>();
        }


        // LK TODO do these need to be here?
        if (jsonData.contains("layers")) {
            this->psfLayers = jsonData["layers"].get<std::vector<int>>();

        } else {
            //throw std::runtime_error("Missing required parameter: secondpsflayers");
        }
        if (jsonData.contains("subimages")) {
            this->psfCubes = jsonData["subimages"].get<std::vector<int>>();

        } else {
            //throw std::runtime_error("Missing required parameter: secondpsfcubes");
        }



        if (jsonData.contains("psfx")) {
            this->sizeX = jsonData.at("psfx").get<int>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfx");
        }
        if (jsonData.contains("psfy")) {
            this->sizeY = jsonData.at("psfy").get<int>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfy");
        }
        if (jsonData.contains("psfz")) {
            this->sizeZ = jsonData.at("psfz").get<int>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfz");
        }

        if (!jsonData.contains("sigmax")){
            if (!jsonData.contains("resolutionx")){
                throw std::runtime_error("[ERROR] Missing required parameter: sigmax or resolutionx");
            }
            this->sigmaX = convertResolution(jsonData.at("resolutionx").get<double>());
        }
        else{this->sigmaX = convertSigma(jsonData.at("sigmax").get<double>());}


        if (!jsonData.contains("sigmay")){
            if (!jsonData.contains("resolutiony")){
                throw std::runtime_error("[ERROR] Missing required parameter: sigmay or resolutiony");
            }
            this->sigmaY = convertResolution(jsonData.at("resolutiony").get<double>());
        }
        else{this->sigmaY = convertSigma(jsonData.at("sigmay").get<double>());}

        if (!jsonData.contains("sigmaz")){
            if (!jsonData.contains("resolutionz")){
                throw std::runtime_error("[ERROR] Missing required parameter: sigmaz or resolutionz");
            }
            this->sigmaZ = convertResolution(jsonData.at("resolutionz").get<double>());
        }
        else{this->sigmaZ = convertSigma(jsonData.at("sigmaz").get<double>());}



    } catch (const json::exception &e) {
        std::cerr << "[ERROR] Invalid PSF JSON structure: " << e.what() << std::endl;
        return false;
    }
    return true;
}



void GaussianPSFConfig::printValues() {
    std::cout << "[INFO] PSF parameters loaded from JSON file:" << std::endl;
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
    return convertSigma(resolution_nm * pixelScaling/nanometerScale);
}

double GaussianPSFConfig::convertSigma(double sigma){
    return sigma * qualityFactor;
}

