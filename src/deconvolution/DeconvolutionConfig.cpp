#include "deconvolution/DeconvolutionConfig.h"


bool DeconvolutionConfig::loadFromJSON(const json& jsonData) {
    // Use readParameterOptional for all parameters
    readParameterOptional<std::string>(jsonData, "algorithmName", algorithmName);
    readParameterOptional<int>(jsonData, "subimageSize", subimageSize);
    readParameterOptional<int>(jsonData, "iterations", iterations);
    readParameterOptional<double>(jsonData, "epsilon", epsilon);
    readParameterOptional<bool>(jsonData, "grid", grid);
    readParameterOptional<double>(jsonData, "lambda", lambda);
    readParameterOptional<int>(jsonData, "borderType", borderType);
    readParameterOptional<int>(jsonData, "psfSafetyBorder", psfSafetyBorder);
    readParameterOptional<int>(jsonData, "cubeSize", cubeSize);
    readParameterOptional<std::vector<int>>(jsonData, "secondpsflayers", secondpsflayers);
    readParameterOptional<std::vector<int>>(jsonData, "secondpsfcubes", secondpsfcubes);
    // readParameterOptional<bool>(jsonData, "time", time);
    // readParameterOptional<bool>(jsonData, "saveSubimages", saveSubimages);
    // readParameterOptional<std::string>(jsonData, "gpu", gpu);
    
    return true;
}

