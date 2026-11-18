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

#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"

#include "dolphin/deconvolution/AlgorithmNames.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"

#include "dolphin/deconvolution/configs/RichardsonLucyConfig.h"
#include "dolphin/deconvolution/configs/ConvolutionConfig.h"
#include "dolphin/deconvolution/configs/InverseFilterConfig.h"
#include "dolphin/deconvolution/configs/RegularizedInverseFilterConfig.h"
#include "dolphin/deconvolution/configs/RichardsonLucyTVConfig.h"
#include "dolphin/deconvolution/configs/RichardsonLucyADConfig.h"
#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"
#include "dolphin/deconvolution/configs/PoissonMLEUnitStepConfig.h"

#include "dolphin/deconvolution/algorithms/ConvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/PoissonMLEDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/PoissonMLEUnitStepDeconvolutionAlgorithm.h"

#include <stdexcept>
#include <spdlog/spdlog.h>

DeconvolutionAlgorithmFactory& DeconvolutionAlgorithmFactory::getInstance() {
    static DeconvolutionAlgorithmFactory instance;
    if (!instance.initialized_) {
        instance.registerAlgorithms();
        instance.initialized_ = true;
    }
    return instance;
}

void DeconvolutionAlgorithmFactory::registerAlgorithm(const std::string& name,
                                                      AlgorithmCreator creator,
                                                      ConfigCreator configCreator) {
    if (algorithms_.find(name) != algorithms_.end()) {
        spdlog::warn("Algorithm '{}' is already registered. Overwriting.", name);
    }
    algorithms_[name] = std::move(creator);
    configs_[name] = std::move(configCreator);
}

std::shared_ptr<DeconvolutionConfig> DeconvolutionAlgorithmFactory::createConfigForAlgorithm(
    const std::string& algorithmName
) const {
    auto it = configs_.find(algorithmName);
    if (it == configs_.end()) {
        throw std::runtime_error("Unknown algorithm (no config registered): " + algorithmName);
    }
    return it->second();
}

std::shared_ptr<DeconvolutionConfig> DeconvolutionAlgorithmFactory::createConfigForAlgorithm(
    const std::string& algorithmName, const json& data
) const {
    auto config = createConfigForAlgorithm(algorithmName);
    if (!config->loadFromJSON(data)) {
        throw std::runtime_error("Failed to parse deconvolution config for algorithm: " + algorithmName);
    }
    return config;
}

std::shared_ptr<DeconvolutionConfig> DeconvolutionAlgorithmFactory::createConfigFromJSON(
    const json& configData
) {
    if (!configData.is_object()) {
        throw std::runtime_error(
            "Deconvolution config must be a JSON object with 'algorithm_name'. "
            "Found type: " + std::string(configData.type_name()));
    }

    std::string algorithmName = DeconvolutionAlgorithms::kRichardsonLucy;
    if (configData.contains("algorithm_name")) {
        algorithmName = configData.at("algorithm_name").get<std::string>();
    }

    return createConfigForAlgorithm(algorithmName, configData);
}

DeconvolutionAlgorithm* DeconvolutionAlgorithmFactory::createAlgorithmFor(
    const DeconvolutionConfig& config
) const {
    auto it = algorithms_.find(config.algorithmName);
    if (it == algorithms_.end()) {
        throw std::runtime_error("Unknown algorithm: " + config.algorithmName);
    }
    // Creators return a raw pointer so callers can pick shared or unique ownership.
    return it->second();
}

std::shared_ptr<DeconvolutionAlgorithm> DeconvolutionAlgorithmFactory::create(
    const DeconvolutionConfig& config
) {
    return createShared(config);
}

std::shared_ptr<DeconvolutionAlgorithm> DeconvolutionAlgorithmFactory::createShared(
    const DeconvolutionConfig& config
) {
    std::shared_ptr<DeconvolutionAlgorithm> algorithm(createAlgorithmFor(config));

    // Configure the algorithm with the provided config
    algorithm->configure(config);
    return algorithm;
}

std::unique_ptr<DeconvolutionAlgorithm> DeconvolutionAlgorithmFactory::createUnique(
    const DeconvolutionConfig& config
) {
    std::unique_ptr<DeconvolutionAlgorithm> algorithm(createAlgorithmFor(config));

    // Configure the algorithm with the provided config
    algorithm->configure(config);
    return algorithm;
}

std::vector<std::string> DeconvolutionAlgorithmFactory::getAvailableAlgorithms() const {
    std::vector<std::string> names;
    names.reserve(algorithms_.size());

    for (const auto& [name, creator] : algorithms_) {
        names.push_back(name);
    }

    return names;
}

bool DeconvolutionAlgorithmFactory::isAlgorithmAvailable(const std::string& name) const {
    return algorithms_.find(name) != algorithms_.end();
}

void DeconvolutionAlgorithmFactory::registerAlgorithms() {

    registerAlgorithm(DeconvolutionAlgorithms::kRichardsonLucy,
        []() -> DeconvolutionAlgorithm* { return new RLDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<RichardsonLucyConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kConvolution,
        []() -> DeconvolutionAlgorithm* { return new ConvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<ConvolutionConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kInverseFilter,
        []() -> DeconvolutionAlgorithm* { return new InverseFilterDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<InverseFilterConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kRichardsonLucyTotalVariation,
        []() -> DeconvolutionAlgorithm* { return new RLTVDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<RichardsonLucyTVConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kRegularizedInverseFilter,
        []() -> DeconvolutionAlgorithm* { return new RegularizedInverseFilterDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<RegularizedInverseFilterConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kRichardsonLucyAdaptiveDamping,
        []() -> DeconvolutionAlgorithm* { return new RLADDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<RichardsonLucyADConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kPoissonMLE,
        []() -> DeconvolutionAlgorithm* { return new PoissonMLEDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<PoissonMLEConfig>(); });

    registerAlgorithm(DeconvolutionAlgorithms::kPoissonMLEUnitStep,
        []() -> DeconvolutionAlgorithm* { return new PoissonMLEUnitStepDeconvolutionAlgorithm(); },
        []() -> std::shared_ptr<DeconvolutionConfig> { return std::make_shared<PoissonMLEUnitStepConfig>(); });
}
