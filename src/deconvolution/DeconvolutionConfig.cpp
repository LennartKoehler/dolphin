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

#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include <spdlog/spdlog.h>

DeconvolutionConfig::DeconvolutionConfig() {
    registerAllParameters();
}

DeconvolutionConfig::DeconvolutionConfig(const DeconvolutionConfig& other)
    : Config(),
    algorithmName(other.algorithmName),
    iterations(other.iterations),
    epsilon(other.epsilon),
    lambda(other.lambda),
    paddingFillType(other.paddingFillType),
    paddingStrategyType(other.paddingStrategyType),
    featheringRadius(other.featheringRadius),
    cubeSize(other.cubeSize),
    cubePadding(other.cubePadding),
    deconvolutionType(other.deconvolutionType)
    {
        registerAllParameters();
    }


DeconvolutionConfig DeconvolutionConfig::createFromJSONFile(const std::string& filePath) {

    json jsonData = loadJSONFile(filePath);
    DeconvolutionConfig deconvConfig;

    // either load from config tag
    if (jsonData.contains("deconvolution_config")){
        deconvConfig.loadFromJSON(jsonData["deconvolution_config"]);
    }
    // or from entire file
    else{
        if (!deconvConfig.loadFromJSON(jsonData)) {
            throw std::runtime_error("Failed to parse config file: " + filePath);
        }
    }

    return deconvConfig;
}

ConfigMap paddingFillTypeMap{{
    {"zero", PaddingFillType::ZERO},
    {"mirror", PaddingFillType::MIRROR},
    {"linear", PaddingFillType::LINEAR},
    {"quadratic", PaddingFillType::QUADRATIC},
    {"sinusoid", PaddingFillType::SINUSOID},
    {"gaussian", PaddingFillType::GAUSSIAN},
}};


ConfigMap paddingStrategyTypeMap{{
    {"none", PaddingStrategyType::NONE},
    {"parent", PaddingStrategyType::PARENT},
    {"full_psf", PaddingStrategyType::FULL_PSF},
    {"manual", PaddingStrategyType::MANUAL},
}};


void DeconvolutionConfig::registerAllParameters() {
    static std::vector<std::string> algorithmOptions =
        DeconvolutionAlgorithmFactory::getInstance().getAvailableAlgorithms();
    static void* algorithmOptionsVoid = static_cast<void*>(&algorithmOptions);

    const void* paddingFillMap_p = static_cast<const void*>(&paddingFillTypeMap);// oh boy
    const void* paddingStrategyMap_p = static_cast<const void*>(&paddingStrategyTypeMap);// oh boy
    // Register each parameter as a ConfigParameter struct
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    parameters.push_back({ParameterType::StringSelection, &algorithmName, "algorithmName", false, "algorithm_name", "-a,--algorithm", "Algorithm selection", true, false, 0.0, 0.0, algorithmOptionsVoid});
    parameters.push_back({ParameterType::Int, &iterations, "iterations", true, "iterations", "--iterations", "Iterations", false, true, 1.0, 10000.0, nullptr});
    parameters.push_back({ParameterType::Float, &epsilon, "epsilon", true, "epsilon", "--epsilon", "Epsilon", false, true, 1e-12, 1e-3, nullptr});
    parameters.push_back({ParameterType::Float, &lambda, "lambda", true, "lambda", "--lambda", "Lambda regularization", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Map, &paddingFillType, "paddingFill", true, "padding_fill", "--paddingFill", "Type of fill method for the image padding", false, true, 0.0, 5.0, paddingFillMap_p});
    parameters.push_back({ParameterType::Map, &paddingStrategyType, "paddingStrategy", true, "padding_strategy", "--paddingStrategy", "paddingStrategy", false, true, 0.0, 5, paddingStrategyMap_p});
    parameters.push_back({ParameterType::Float, &paddingRelativeMax, "paddingRelativeMax", true, "padding_relative_max", "--paddingRelativeMax",
        "Pad the image up until the PSF is below this Value * Max value of PSF", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Int, &featheringRadius, "featheringRadius", true, "feathering_radius", "--featheringRadius", "Enable featheringRadius", false, false, 0.0, 100000.0, nullptr});
    parameters.push_back({ParameterType::IntArray3, &cubeSize, "cubeSize", false, "cube_size", "--cubeSize", "Size of the cube used (x,y,z)", false, false, 0.0, 0.0, nullptr, 3});
    parameters.push_back({ParameterType::IntArray3, &cubePadding, "cubePadding", false, "cube_padding", "--cubePadding", "Padding for each cube (x,y,z)", false, false, 0.0, 0.0, nullptr, 3});
    parameters.push_back({ParameterType::String, &deconvolutionType, "deconvolutionType", true, "deconvolution_type", "--deconvolutionType", "Deconvolution strategy type", false, false, 0.0, 0.0, nullptr});

}
