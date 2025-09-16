#include "deconvolution/DeconvolutionConfig.h"

DeconvolutionConfig::DeconvolutionConfig() {
    registerAllParameters();
}

void DeconvolutionConfig::registerAllParameters(){
    bool optional = true;
    
    // Register all deconvolution parameters
    registerParameter("algorithmName", algorithmName, !optional);  // Required
    registerParameter("subimageSize", subimageSize, optional);
    registerParameter("iterations", iterations, optional);
    registerParameter("epsilon", epsilon, optional);
    registerParameter("grid", grid, optional);
    registerParameter("lambda", lambda, optional);
    registerParameter("borderType", borderType, optional);
    registerParameter("psfSafetyBorder", psfSafetyBorder, optional);
    registerParameter("cubeSize", cubeSize, optional);
    registerRangeMap("layerPSFMap", layerPSFMap, !optional);
    registerRangeMap("cubePSFMap", cubePSFMap, !optional);
    

    
    // Commented out parameters - add if needed
    // registerParameter("time", time, optional);
    // registerParameter("saveSubimages", saveSubimages, optional);
    // registerParameter("gpu", gpu, optional);
}
DeconvolutionConfig::DeconvolutionConfig(const DeconvolutionConfig& other)
    : Config(other),
    algorithmName(other.algorithmName),
    subimageSize(other.subimageSize),
    iterations(other.iterations),
    epsilon(other.epsilon),
    grid(other.grid),
    lambda(other.lambda),
    borderType(other.borderType),
    psfSafetyBorder(other.psfSafetyBorder),
    cubeSize(other.cubeSize),
    layerPSFMap(other.layerPSFMap),
    cubePSFMap(other.cubePSFMap){
    registerAllParameters();
}

void DeconvolutionConfig::registerRangeMap(const std::string& jsonTag, RangeMap<std::string>& field, bool optional){
    ReadWriteHelper param;

    param.jsonTag = jsonTag;
    param.reader = [&field, jsonTag](const json& jsonData) {
        if (jsonData.contains(jsonTag)) {
            const json& subJson = jsonData.at(jsonTag);
            field.loadFromJSON(subJson);
        }
        else {
            field.clear();
            std::cout << "[INFO] No deconvolution parameters found, running with default parameters" << std::endl;
        }
    };
    
    // Writer lambda
    param.writer = [&field, jsonTag](ordered_json& jsonData) {
        // Convert RangeMap to JSON format
        ordered_json rangeMapJson = ordered_json::object();
        
        // Iterate through the RangeMap and convert to JSON
        for (const auto& [index, values] : field) {
            if (!values.empty()) {
                // Convert single index to range format for JSON
                std::string rangeKey = std::to_string(index) + ":" + std::to_string(index + 1);
                rangeMapJson[rangeKey] = values;
            }
        }
        
        jsonData[jsonTag] = rangeMapJson;
    };
    
    parameters.push_back(std::move(param));
}