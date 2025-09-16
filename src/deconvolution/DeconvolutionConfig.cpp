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

void DeconvolutionConfig::registerRangeMap(const std::string& jsonTag, RangeMap<std::string>& field, bool optional) {
    ReadWriteHelper param;

    param.jsonTag = jsonTag;
    
    // Reader lambda - captures field by reference
    param.reader = [&field, jsonTag](const json& jsonData) {
        if (jsonData.contains(jsonTag)) {
            try {
                const json& subJson = jsonData.at(jsonTag);
                field.loadFromJSON(subJson);
                std::cout << "[INFO] Loaded " << jsonTag << " configuration" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to load " << jsonTag << ": " << e.what() << std::endl;
                field.clear(); // Clear field on error
            }
        } else {
            std::cout << "[INFO] No " << jsonTag << " found, using defaults" << std::endl;
            field.clear(); // Ensure field is empty if not found
        }
    };
    
    // Writer lambda - captures field by reference
    param.writer = [&field, jsonTag](ordered_json& jsonData) {
        // Convert RangeMap to JSON format
        ordered_json rangeMapJson = ordered_json::object();
        
        // Iterate through the RangeMap ranges
        for (const auto& range : field) {
            if (!range.values.empty()) {
                // Convert range to string format for JSON
                std::string rangeKey;
                if (range.end == -1) {
                    rangeKey = std::to_string(range.start) + ":";
                } else {
                    rangeKey = std::to_string(range.start) + ":" + std::to_string(range.end);
                }
                rangeMapJson[rangeKey] = range.values;
            }
        }
        
        jsonData[jsonTag] = rangeMapJson;
    };
    
    parameters.push_back(std::move(param));
}