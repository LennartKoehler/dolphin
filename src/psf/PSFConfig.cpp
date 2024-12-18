#include "PSFConfig.h"
#include <fstream>
#include <iostream>

#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

bool PSFConfig::loadFromJSON(const std::string &filePath) {
    //TODO put stuff from main.cpp here, load every PSF parameter in PSFConfig object

    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("[ERROR] Could not open file: " + filePath);
    }

    json j;
    inputFile >> j;
    // JSON-Parameter lesen und in Variablen speichern
    try {

        if (j.contains("layers")) {
            this->psfLayers = j["layers"].get<std::vector<int>>();

        } else {
            //throw std::runtime_error("Missing required parameter: secondpsflayers");
        }
        if (j.contains("subimages")) {
            this->psfCubes = j["subimages"].get<std::vector<int>>();

        } else {
            //throw std::runtime_error("Missing required parameter: secondpsfcubes");
        }
        if (j.contains("path")) {
            this->psfPath = j["path"].get<std::string>();
            return true;
        }


        if (j.contains("psfx")) {
            this->x = j.at("psfx").get<int>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfx");
        }
        if (j.contains("psfy")) {
            this->y = j.at("psfy").get<int>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfy");
        }
        if (j.contains("psfz")) {
            this->z = j.at("psfz").get<int>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfz");
        }
        if (j.contains("sigmax")) {
            this->sigmax = j.at("sigmax").get<double>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: sigmax");
        }
        if (j.contains("sigmay")) {
            this->sigmay = j.at("sigmay").get<double>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: sigmay");
        }
        if (j.contains("sigmaz")) {
            this->sigmaz = j.at("sigmaz").get<double>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: sigmaz");
        }

        if (j.contains("psfmodel")) {
            this->psfModel = j.value("psfmodel", "gauss");
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: psfModel");
        }


    } catch (const json::exception &e) {
        std::cerr << "[ERROR] Invalid PSF JSON structure: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool PSFConfig::compareDim(const PSFConfig &other) {
    if(this->x != other.x || this->y != other.y || this->z != other.z) {
        std::cerr << "[ERROR] All PSFs have to be the same size" << std::endl;
        return false;
    }
}

void PSFConfig::printValues() {
    std::cout << "[INFO] PSF parameters loaded from JSON file:" << std::endl;
    std::cout << "  sigmax: " << this->sigmax << ", sigmay: " << this->sigmay
              << ", sigmaz: " << this->sigmaz << std::endl;
    std::cout << "  psfx: " << this->x << ", psfy: " << this->y
              << ", psfz: " << this->z << std::endl;
    std::cout << "  psfmodel: " << psfModel << std::endl;


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

