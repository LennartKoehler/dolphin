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
    borderType(other.borderType),
    featheringRadius(other.featheringRadius)
    {
        registerAllParameters();
    }




void DeconvolutionConfig::registerAllParameters() {
    static std::vector<std::string> algorithmOptions =
        DeconvolutionAlgorithmFactory::getInstance().getAvailableAlgorithms();
    static void* algorithmOptionsVoid = static_cast<void*>(&algorithmOptions);
    
    // Register each parameter as a ConfigParameter struct
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    parameters.push_back({ParameterType::VectorString, &algorithmName, "algorithmName", false, "algorithmName", "-a,--algorithm", "Algorithm selection", true, false, 0.0, 0.0, algorithmOptionsVoid});
    parameters.push_back({ParameterType::Int, &iterations, "iterations", true, "iterations", "--iterations", "Iterations", false, true, 1.0, 10000.0, nullptr});
    parameters.push_back({ParameterType::Float, &epsilon, "epsilon", true, "epsilon", "--epsilon", "Epsilon", false, true, 1e-12, 1e-3, nullptr});
    parameters.push_back({ParameterType::Float, &lambda, "lambda", true, "lambda", "--lambda", "Lambda regularization", false, false, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Int, &borderType, "borderType", true, "borderType", "--borderType", "Border type", false, true, 0.0, 5.0, nullptr});
    parameters.push_back({ParameterType::Int, &featheringRadius, "featheringRadius", true, "featheringRadius", "--featheringRadius", "Enable featheringRadius", false, false, 0.0, 100000.0, nullptr});
    
}
