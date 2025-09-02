#include "DeconvolutionConfig.h"


bool DeconvolutionConfig::loadFromJSON(const json& jsonData) {
    algorithmName = readParameter<std::string>(jsonData, "algorithm");
    epsilon = readParameter<double>(jsonData, "epsilon");
    iterations = readParameter<int>(jsonData, "iterations");
    lambda = readParameter<double>(jsonData, "lambda");
    psfSafetyBorder = readParameter<int>(jsonData, "psfSafetyBorder");
    subimageSize = readParameter<int>(jsonData, "subimageSize");
    borderType = readParameter<int>(jsonData, "borderType");
    grid = readParameter<bool>(jsonData, "grid");
}

