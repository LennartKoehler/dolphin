#include "PSFConfig.h"
#include <fstream>
#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

void PSFConfig::loadFromJSON(const std::string &filePath) {
    //TODO put stuff from main.cpp here, load every PSF parameter in PSFConfig object

    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    json j;
    inputFile >> j;

    if (j.contains("iterations")) {
        this->x = j.at("psfx").get<int>();
    } else {
        throw std::runtime_error("Missing required parameter: psfx");
    }
}

