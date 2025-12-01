/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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
    maxMem_GB(other.maxMem_GB),
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
    parameters.push_back({ParameterType::Float, &maxMem_GB, "maxMem_GB", false, "maxMem_GB", "--maxMem_GB", "Maximum memory usage", false, false, 0.0, 0.0, nullptr});
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

    

//------------------------------------------------


// LabeledDeconvolutionConfig::LabeledDeconvolutionConfig() 
//     : DeconvolutionConfig() {
//     deconvolutionType = "labeled";
//     // Base class already calls registerAllParameters()
// }

// LabeledDeconvolutionConfig::LabeledDeconvolutionConfig(const LabeledDeconvolutionConfig& other)
//     : DeconvolutionConfig(other),
//       labelPSFMap(other.labelPSFMap) {
//     // Base class copy constructor already handles parameter registration
// }

// bool LabeledDeconvolutionConfig::loadFromJSON(const json& jsonData) {
//     // First load base class parameters
//     bool success = DeconvolutionConfig::loadFromJSON(jsonData);

//     // Handle additional labelPSFMap parameter
//     try {
//         if (jsonData.contains("labelPSFMap")) {
//             labelPSFMap.loadFromJSON(jsonData.at("labelPSFMap"));
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "Failed to load labelPSFMap parameter: " << e.what() << std::endl;
//         success = false;
//     }
    
//     return success;
// }

// json LabeledDeconvolutionConfig::writeToJSON() {
//     // Get base class JSON data
//     json jsonData = DeconvolutionConfig::writeToJSON();
    
//     // Add labelPSFMap to JSON
//     ordered_json labelPSFJson = ordered_json::object();
//     for (const auto& range : labelPSFMap) {
//         if (!range.values.empty()) {
//             std::string rangeKey;
//             if (range.end == -1) {
//                 rangeKey = std::to_string(range.start) + ":";
//             } else {
//                 rangeKey = std::to_string(range.start) + ":" + std::to_string(range.end);
//             }
//             labelPSFJson[rangeKey] = range.values;
//         }
//     }
//     if (!labelPSFJson.empty()) {
//         jsonData["labelPSFMap"] = labelPSFJson;
//     }
    
//     return jsonData;
// }
