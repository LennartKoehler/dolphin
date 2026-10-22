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
    paddingRelativeMax(other.paddingRelativeMax),
    featheringRadius(other.featheringRadius),
    cubeSize(other.cubeSize),
    cubePadding(other.cubePadding),
    poissonSnr(other.poissonSnr),
    poissonLambda(other.poissonLambda),
    poissonBackground(other.poissonBackground),
    poissonSieve(other.poissonSieve),
    poissonPrior(other.poissonPrior),
    poissonSievePasses(other.poissonSievePasses)
    {
        parameters.clear();
        registerAllParameters();
    }

DeconvolutionConfig& DeconvolutionConfig::operator=(const DeconvolutionConfig& other) {
    if (this != &other) {
        algorithmName = other.algorithmName;
        iterations = other.iterations;
        epsilon = other.epsilon;
        lambda = other.lambda;
        paddingFillType = other.paddingFillType;
        paddingStrategyType = other.paddingStrategyType;
        paddingRelativeMax = other.paddingRelativeMax;
        featheringRadius = other.featheringRadius;
        cubeSize = other.cubeSize;
        cubePadding = other.cubePadding;
        poissonSnr = other.poissonSnr;
        poissonLambda = other.poissonLambda;
        poissonBackground = other.poissonBackground;
        poissonSieve = other.poissonSieve;
        poissonPrior = other.poissonPrior;
        poissonSievePasses = other.poissonSievePasses;
        parameters.clear();
        registerAllParameters();
    }
    return *this;
}

DeconvolutionConfig::DeconvolutionConfig(DeconvolutionConfig&& other) noexcept
    : Config(),
    algorithmName(std::move(other.algorithmName)),
    iterations(other.iterations),
    epsilon(other.epsilon),
    lambda(other.lambda),
    paddingFillType(other.paddingFillType),
    paddingStrategyType(other.paddingStrategyType),
    paddingRelativeMax(other.paddingRelativeMax),
    featheringRadius(other.featheringRadius),
    cubeSize(other.cubeSize),
    cubePadding(other.cubePadding),
    poissonSnr(other.poissonSnr),
    poissonLambda(other.poissonLambda),
    poissonBackground(other.poissonBackground),
    poissonSieve(other.poissonSieve),
    poissonPrior(other.poissonPrior),
    poissonSievePasses(other.poissonSievePasses)
    {
        parameters.clear();
        registerAllParameters();
    }

DeconvolutionConfig& DeconvolutionConfig::operator=(DeconvolutionConfig&& other) noexcept {
    if (this != &other) {
        algorithmName = std::move(other.algorithmName);
        iterations = other.iterations;
        epsilon = other.epsilon;
        lambda = other.lambda;
        paddingFillType = other.paddingFillType;
        paddingStrategyType = other.paddingStrategyType;
        paddingRelativeMax = other.paddingRelativeMax;
        featheringRadius = other.featheringRadius;
        cubeSize = other.cubeSize;
        cubePadding = other.cubePadding;
        poissonSnr = other.poissonSnr;
        poissonLambda = other.poissonLambda;
        poissonBackground = other.poissonBackground;
        poissonSieve = other.poissonSieve;
        poissonPrior = other.poissonPrior;
        poissonSievePasses = other.poissonSievePasses;
        parameters.clear();
        registerAllParameters();
    }
    return *this;
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
    parameters.push_back({ParameterType::StringSelection, &algorithmName, "Algorithm Name", true, "algorithm_name", "--algorithm_name", "Algorithm selection", false, false, 0.0, 0.0, algorithmOptionsVoid});
    parameters.push_back({ParameterType::Int, &iterations, "Iterations", true, "iterations", "--iterations", "Iterations", false, true, 1.0, 10000.0, nullptr});
    parameters.push_back({ParameterType::Float, &epsilon, "Epsilon", true, "epsilon", "--epsilon", "Epsilon", false, true, 1e-12, 1e-3, nullptr});
    parameters.push_back({ParameterType::Float, &lambda, "Lambda", true, "lambda", "--lambda", "Lambda regularization", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Map, &paddingFillType, "Padding Fill", true, "padding_fill", "--padding_fill", "Type of fill method for the image padding", false, true, 0.0, 5.0, paddingFillMap_p});
    parameters.push_back({ParameterType::Map, &paddingStrategyType, "Padding Strategy", true, "padding_strategy", "--padding_strategy", "paddingStrategy", false, true, 0.0, 5, paddingStrategyMap_p});
    parameters.push_back({ParameterType::Float, &paddingRelativeMax, "Padding Relative Max", true, "padding_relative_max", "--padding_relative_max",
        "Pad the image up until the PSF is below this Value * Max value of PSF", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Int, &featheringRadius, "Feathering Radius", true, "feathering_radius", "--feathering_radius", "Enable featheringRadius", false, false, 0.0, 100000.0, nullptr});
    parameters.push_back({ParameterType::IntArray3, &cubeSize, "Cube Size", true, "cube_size", "--cube_size", "Size of the cube used (x,y,z)", false, false, 0.0, 0.0, nullptr, 3});
    parameters.push_back({ParameterType::IntArray3, &cubePadding, "Cube Padding", true, "cube_padding", "--cube_padding", "Padding for each cube (x,y,z)", false, false, 0.0, 0.0, nullptr, 3});
    // PoissonMLE (Poisson / I-divergence likelihood) specific parameters
    parameters.push_back({ParameterType::Float, &poissonSnr, "Poisson SNR", true, "poisson_snr", "--poisson_snr", "Effective SNR / acuity master dial for PoissonMLE", false, true, 1e-3, 1e5, nullptr});
    parameters.push_back({ParameterType::Float, &poissonLambda, "Poisson Lambda", true, "poisson_lambda", "--poisson_lambda", "Shrinkage strength for PoissonMLE prior step (0 = derive from SNR)", false, true, 0.0, 1e4, nullptr});
    parameters.push_back({ParameterType::Float, &poissonBackground, "Poisson Background", true, "poisson_background", "--poisson_background", "Background level for PoissonMLE (0 = estimate from data)", false, true, 0.0, 1e8, nullptr});
    parameters.push_back({ParameterType::Bool, &poissonSieve, "Poisson Sieve", true, "poisson_sieve", "--poisson_sieve", "Enable annealed anisotropic Gaussian smoothing (sieve) in PoissonMLE", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &poissonPrior, "Poisson Prior", true, "poisson_prior", "--poisson_prior", "Enable lambda-shrinkage + flux conservation prior step in PoissonMLE", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Int, &poissonSievePasses, "Poisson Sieve Passes", true, "poisson_sieve_passes", "--poisson_sieve_passes", "Number of sieve smoothing passes per iteration in PoissonMLE", false, true, 0.0, 64.0, nullptr});
    // its labeled if the labeled image is given
    // parameters.push_back({ParameterType::String, &deconvolutionType, "Deconvolution Type", true, "deconvolution_type", "--deconvolution_type", "Deconvolution strategy type", false, false, 0.0, 0.0, nullptr});

}
