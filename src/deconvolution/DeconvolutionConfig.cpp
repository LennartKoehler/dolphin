#include "deconvolution/DeconvolutionConfig.h"

DeconvolutionConfig::DeconvolutionConfig() {
    registerAllParameters();
}

DeconvolutionConfig::DeconvolutionConfig(const DeconvolutionConfig& other)
    : Config(other),
    algorithmName(other.algorithmName),
    subimageSize(other.subimageSize),
    iterations(other.iterations),
    epsilon(other.epsilon),
    lambda(other.lambda),
    borderType(other.borderType),
    backenddeconv(other.backenddeconv),
    nThreads(other.nThreads),
    maxMemGB(other.maxMemGB),
    verbose(other.verbose),
    layerPSFMap(other.layerPSFMap),
    cubePSFMap(other.cubePSFMap)
    {
        registerAllParameters();
    }



bool DeconvolutionConfig::loadFromJSON(const json& jsonData) {
    bool success = Config::loadFromJSON(jsonData);

    // Handle RangeMap parameters separately since they're not in the base Config system yet
    try {
        if (jsonData.contains("cubePSFMap")) {
            cubePSFMap.loadFromJSON(jsonData.at("cubePSFMap"));
        }
        if (jsonData.contains("layerPSFMap")) {
            layerPSFMap.loadFromJSON(jsonData.at("layerPSFMap"));
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load RangeMap parameters: " << e.what() << std::endl;
        success = false;
    }
    
    return success;
}

void DeconvolutionConfig::registerAllParameters() {
    static std::vector<std::string> algorithmOptions =
        DeconvolutionAlgorithmFactory::getInstance().getAvailableAlgorithms();
    static std::vector<std::string> backendOptions =
        BackendFactory::getInstance().getAvailableBackends();

    static void* algorithmOptionsVoid = static_cast<void*>(&algorithmOptions);
    static void* backendOptionsVoid = static_cast<void*>(&backendOptions);

    // Register each parameter as a ConfigParameter struct
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    parameters.push_back({ParameterType::VectorString, &algorithmName, "algorithmName", false, "algorithmName", "-a,--algorithm", "Algorithm selection", true, false, 0.0, 0.0, algorithmOptionsVoid});
    parameters.push_back({ParameterType::Int, &subimageSize, "subimageSize", true, "subimageSize", "--subimageSize", "CubeSize/EdgeLength", false, true, 0.0, 10000.0, nullptr});
    parameters.push_back({ParameterType::Int, &iterations, "iterations", true, "iterations", "--iterations", "Iterations", false, true, 1.0, 10000.0, nullptr});
    parameters.push_back({ParameterType::Float, &epsilon, "epsilon", true, "epsilon", "--epsilon", "Epsilon", false, true, 1e-12, 1e-3, nullptr});
    parameters.push_back({ParameterType::Float, &lambda, "lambda", true, "lambda", "--lambda", "Lambda regularization", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Int, &borderType, "borderType", true, "borderType", "--borderType", "Border type", false, true, 0.0, 5.0, nullptr});
    parameters.push_back({ParameterType::VectorString, &backenddeconv, "backenddeconv", true, "backenddeconv", "--backenddeconv", "Backend type", false, false, 0.0, 0.0, backendOptionsVoid});
    parameters.push_back({ParameterType::Int, &nThreads, "nThreads", false, "nThreads", "--nThreads", "Number of threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Float, &maxMemGB, "maxMemGB", false, "maxMemGB", "--maxMemGB", "Maximum memory usage", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &verbose, "verbose", true, "verbose", "--verbose", "Enable verbose", false, false, 0.0, 1.0, nullptr});

    // Note: RangeMap parameters are not yet supported in the base Config parameter system
}

json DeconvolutionConfig::writeToJSON() {
    json jsonData = Config::writeToJSON();
    
    // Handle RangeMap parameters separately since they're not in the base Config system yet
    // Convert RangeMap to JSON format for cubePSFMap
    ordered_json cubePSFJson = ordered_json::object();
    for (const auto& range : cubePSFMap) {
        if (!range.values.empty()) {
            std::string rangeKey;
            if (range.end == -1) {
                rangeKey = std::to_string(range.start) + ":";
            } else {
                rangeKey = std::to_string(range.start) + ":" + std::to_string(range.end);
            }
            cubePSFJson[rangeKey] = range.values;
        }
    }
    if (!cubePSFJson.empty()) {
        jsonData["cubePSFMap"] = cubePSFJson;
    }
    
    // Convert RangeMap to JSON format for layerPSFMap
    ordered_json layerPSFJson = ordered_json::object();
    for (const auto& range : layerPSFMap) {
        if (!range.values.empty()) {
            std::string rangeKey;
            if (range.end == -1) {
                rangeKey = std::to_string(range.start) + ":";
            } else {
                rangeKey = std::to_string(range.start) + ":" + std::to_string(range.end);
            }
            layerPSFJson[rangeKey] = range.values;
        }
    }
    if (!layerPSFJson.empty()) {
        jsonData["layerPSFMap"] = layerPSFJson;
    }
    
    return jsonData;
}

    
//     // Writer lambda - captures field by reference
//     param.writer = [&field, jsonTag](ordered_json& jsonData) {
//         // Convert RangeMap to JSON format
//         ordered_json rangeMapJson = ordered_json::object();
        
//         // Iterate through the RangeMap ranges
//         for (const auto& range : field) {
//             if (!range.values.empty()) {
//                 // Convert range to string format for JSON
//                 std::string rangeKey;
//                 if (range.end == -1) {
//                     rangeKey = std::to_string(range.start) + ":";
//                 } else {
//                     rangeKey = std::to_string(range.start) + ":" + std::to_string(range.end);
//                 }
//                 rangeMapJson[rangeKey] = range.values;
//             }
//         }
        
//         jsonData[jsonTag] = rangeMapJson;
//     };
    
// }