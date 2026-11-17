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

#pragma once

#include <memory>
#include <utility>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Forward declarations
class DeconvolutionAlgorithm;
class DeconvolutionConfig;

/**
 * @brief Factory for creating deconvolution algorithm instances and their configs.
 *
 * Every algorithm is registered together with a creator for its config class, so
 * the config an algorithm consumes is exactly the one that was built for it (the
 * PSFConfig / PSFGenerator pairing). The algorithm name is the key for both.
 *
 * Usage:
 *   auto config = DeconvolutionAlgorithmFactory::getInstance().createConfigFromJSON(json);   // finished
 *   auto algorithm = DeconvolutionAlgorithmFactory::getInstance().create(config);             // configured
 */
class DeconvolutionAlgorithmFactory {
public:
    using AlgorithmCreator = std::function<DeconvolutionAlgorithm*()>;
    using ConfigCreator = std::function<std::shared_ptr<DeconvolutionConfig>()>;

    /**
     * @brief Get the singleton instance of the factory.
     * @return Reference to the factory instance
     */
    static DeconvolutionAlgorithmFactory& getInstance();

    /**
     * @brief Register a new algorithm and the config class it consumes.
     * @param name Algorithm name identifier
     * @param creator Function that creates algorithm instances
     * @param configCreator Function that creates the matching (empty) config
     */
    void registerAlgorithm(const std::string& name, AlgorithmCreator creator, ConfigCreator configCreator);

    /**
     * @brief Create an empty config for the given algorithm name.
     *
     * For callers that bind values later (CLI option binding, GUI defaults).
     * @throws std::runtime_error if no config is registered for the name
     */
    std::shared_ptr<DeconvolutionConfig> createConfigForAlgorithm(const std::string& algorithmName) const;

    /**
     * @brief Create a finished config for a known algorithm name.
     *
     * Mirrors PSFGeneratorFactory::createConfig: the factory creates the config
     * class registered for `algorithmName`, loads all parameters from `data`
     * into it, and returns it ready to use. Callers never call loadFromJSON
     * themselves.
     * @throws std::runtime_error on an unknown algorithm or invalid parameters
     */
    std::shared_ptr<DeconvolutionConfig> createConfigForAlgorithm(const std::string& algorithmName, const json& data) const;

    /**
     * @brief Create a finished config from JSON.
     *
     * The algorithm is selected by the "algorithm_name" field
     * (default: RichardsonLucy); the returned config is fully loaded.
     * @throws std::runtime_error on an unknown algorithm or invalid parameters
     */
    std::shared_ptr<DeconvolutionConfig> createConfigFromJSON(const json& configData);

    /**
     * @brief Create and configure an algorithm for the given config.
     * @throws std::runtime_error if the config's algorithm is unknown or if the
     *         config does not match the algorithm
     */
    std::shared_ptr<DeconvolutionAlgorithm> create(const DeconvolutionConfig& config);
    std::shared_ptr<DeconvolutionAlgorithm> createShared(const DeconvolutionConfig& config);
    std::unique_ptr<DeconvolutionAlgorithm> createUnique(const DeconvolutionConfig& config);

    /**
     * @brief Get list of all available algorithm names.
     */
    std::vector<std::string> getAvailableAlgorithms() const;

    /**
     * @brief Check if an algorithm is available.
     */
    bool isAlgorithmAvailable(const std::string& name) const;

private:
    DeconvolutionAlgorithmFactory() = default;
    ~DeconvolutionAlgorithmFactory() = default;

    // Prevent copying
    DeconvolutionAlgorithmFactory(const DeconvolutionAlgorithmFactory&) = delete;
    DeconvolutionAlgorithmFactory& operator=(const DeconvolutionAlgorithmFactory&) = delete;

    /**
     * @brief Register all available algorithms and their configs.
     * Called automatically on first access to getInstance().
     */
    void registerAlgorithms();

    DeconvolutionAlgorithm* createAlgorithmFor(const DeconvolutionConfig& config) const;

    bool initialized_ = false;
    std::unordered_map<std::string, AlgorithmCreator> algorithms_;
    std::unordered_map<std::string, ConfigCreator> configs_;
};
