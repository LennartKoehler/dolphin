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

DeconvolutionConfig::DeconvolutionConfig() {
    registerAllParameters();
}

DeconvolutionConfig::DeconvolutionConfig(const DeconvolutionConfig& other) {
    copyBaseFields(other);
    paramters.clear();
    registerAllParameters();
}

DeconvolutionConfig& DeconvolutionConfig::operator=(const DeconvolutionConfig& other) {
    if (this != &other) {
        assignBase(other);
        parameters.clear();
        registerAllParameters();
    }
    return *this;
}

DeconvolutionConfig::DeconvolutionConfig(DeconvolutionConfig&& other) noexcept {
    moveBaseFields(other);
    parameters.clear();
    registerAllParameters();
}

DeconvolutionConfig& DeconvolutionConfig::operator=(DeconvolutionConfig&& other) noexcept {
    if (this != &other) {
        moveBaseFields(other);
        parameters.clear();
        registerAllParameters();
    }
    return *this;
}

void DeconvolutionConfig::copyBaseFields(const DeconvolutionConfig& other) {
    algorithmName = other.algorithmName;
    iterations = other.iterations;
    minDenominator = other.minDenominator;
    signalToNoise = other.signalToNoise;
    paddingFillType = other.paddingFillType;
    paddingStrategyType = other.paddingStrategyType;
    paddingRelativeMax = other.paddingRelativeMax;
    featheringRadius = other.featheringRadius;
    cubeSize = other.cubeSize;
    cubePadding = other.cubePadding;
    sliceAllDimensions = other.sliceAllDimensions;
}

void DeconvolutionConfig::moveBaseFields(DeconvolutionConfig& other) {
    algorithmName = std::move(other.algorithmName);
    iterations = other.iterations;
    minDenominator = other.minDenominator;
    signalToNoise = other.signalToNoise;
    paddingFillType = other.paddingFillType;
    paddingStrategyType = other.paddingStrategyType;
    paddingRelativeMax = other.paddingRelativeMax;
    featheringRadius = other.featheringRadius;
    cubeSize = other.cubeSize;
    cubePadding = other.cubePadding;
    sliceAllDimensions = other.sliceAllDimensions;
}

void DeconvolutionConfig::assignBase(const DeconvolutionConfig& other) {
    // Only copies fields: the caller re-registers its parameters afterwards
    // (registerAllParameters() starts from an empty list).
    copyBaseFields(other);
}

std::shared_ptr<DeconvolutionConfig> DeconvolutionConfig::createFromJSON(const json& fileData) {
    // Either load from the "deconvolution_config" tag or from the entire file.
    const json& data = fileData.contains("deconvolution_config")
                           ? fileData.at("deconvolution_config")
                           : fileData;
    if (!data.is_object()) {
        throw std::runtime_error("Deconvolution config must be a JSON object");
    }

    // The factory reads algorithm_name (default RichardsonLucy), creates the
    // matching config class and loads the parameters into it.
    return DeconvolutionAlgorithmFactory::getInstance().createConfigFromJSON(data);
}

std::shared_ptr<DeconvolutionConfig> DeconvolutionConfig::createFromJSONFile(const std::string& filePath) {
    json jsonData = loadJSONFile(filePath);
    return createFromJSON(jsonData);
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
    {"default", PaddingStrategyType::DEFAULT},
    {"none", PaddingStrategyType::NONE},
    {"low", PaddingStrategyType::LOW},
    {"medium", PaddingStrategyType::MEDIUM},
    {"max", PaddingStrategyType::MAX}
}};


void DeconvolutionConfig::registerAllParameters() {
    // Derived constructors call this again after the base constructor registered
    // the shared parameters; start from a clean list so nothing is added twice.
    parameters.clear();

    static std::vector<std::string> algorithmOptions = DeconvolutionAlgorithms::names();
    static void* algorithmOptionsVoid = static_cast<void*>(&algorithmOptions);

    const void* paddingFillMap_p = static_cast<const void*>(&paddingFillTypeMap);// oh boy
    const void* paddingStrategyMap_p = static_cast<const void*>(&paddingStrategyTypeMap);// oh boy
    // Register each parameter as a ConfigParameter struct
    parameters.push_back({ParameterType::StringSelection, &algorithmName, "Algorithm Name", true, "algorithm_name", "--algorithm_name", "Algorithm selection", false, false, 0.0, 0.0, algorithmOptionsVoid});
    parameters.push_back({ParameterType::Int, &iterations, "Iterations", true, "iterations", "--iterations", "Iterations", false, true, 1.0, 10000.0, nullptr});
    parameters.push_back({ParameterType::Float, &minDenominator, "Min Denominator", true, "min_denominator", "--min_denominator", "Min denominator", false, true, 1e-12, 1e-3, nullptr});
    parameters.push_back({ParameterType::Float, &signalToNoise, "Signal to Noise", true, "signal_to_noise", "--signal_to_noise", "Signal to noise ratio", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Map, &paddingFillType, "Padding Fill", true, "padding_fill", "--padding_fill", "Type of fill method for the image padding", false, true, 0.0, 5.0, paddingFillMap_p});
    parameters.push_back({ParameterType::Map, &paddingStrategyType, "Padding Strategy", true, "padding_strategy", "--padding_strategy",
        "Preset PSF padding threshold: none, low, medium, max. Overridden if cube_padding or padding_relative_max is set", false, true, 0.0, 4, paddingStrategyMap_p});
    parameters.push_back({ParameterType::Float, &paddingRelativeMax, "Padding Relative Max", true, "padding_relative_max", "--padding_relative_max",
        "Pad until PSF is below this fraction of its max value (0-1). > 0 overrides padding_strategy", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Int, &featheringRadius, "Feathering Radius", true, "feathering_radius", "--feathering_radius", "Enable featheringRadius", false, false, 0.0, 100000.0, nullptr});
    parameters.push_back({ParameterType::IntArray3, &cubeSize, "Cube Size", true, "cube_size", "--cube_size", "Size of the cube used (x,y,z)", false, false, 0.0, 0.0, nullptr, 3});
    parameters.push_back({ParameterType::IntArray3, &cubePadding, "Cube Padding", true, "cube_padding", "--cube_padding", "Padding for each cube (x,y,z)", false, false, 0.0, 0.0, nullptr, 3});
    parameters.push_back({ParameterType::Bool, &sliceAllDimensions, "Slice all dimensions", true, "slice_all_dimensions", "--slice_all_dimensions", "Slice all dimensions not only Z dimension", false, false, 0.0, 0.0, nullptr});
    // its labeled if the labeled image is given
    // parameters.push_back({ParameterType::String, &deconvolutionType, "Deconvolution Type", true, "deconvolution_type", "--deconvolution_type", "Deconvolution strategy type", false, false, 0.0, 0.0, nullptr});
}
