#include "DeconvolutionConfig.h"
#include <fstream>
#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

void DeconvolutionConfig::loadFromJSON(const std::string &filePath) {
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    json j;
    inputFile >> j;

    if (j.contains("iterations")) {
        this->iterations = j.at("iterations").get<int>();
    } else {
        throw std::runtime_error("Missing required parameter: iterations");
    }
}

