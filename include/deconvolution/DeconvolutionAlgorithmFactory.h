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

// Forward declarations
class DeconvolutionAlgorithm;
class DeconvolutionConfig;

/**
 * @brief Factory class for creating deconvolution algorithm instances with CPU/GPU variants.
 * 
 * This factory supports both CPU and GPU variants of algorithms, with conditional 
 * compilation to ensure GPU variants are only registered when CUDA is available.
 */
class DeconvolutionAlgorithmFactory {
public:
    using AlgorithmCreator = std::function<std::shared_ptr<DeconvolutionAlgorithm>()>;

    /**
     * @brief Get the singleton instance of the factory.
     * @return Reference to the factory instance
     */
    static DeconvolutionAlgorithmFactory& getInstance();

    /**
     * @brief Register a new algorithm with the factory.
     * @param name Algorithm name identifier
     * @param creator Function that creates algorithm instances
     */
    void registerAlgorithm(const std::string& name, AlgorithmCreator creator);

    /**
     * @brief Create an algorithm instance based on configuration.
     * @param config Deconvolution configuration containing algorithm selection
     * @return Shared pointer to the created algorithm instance
     * @throws std::runtime_error if algorithm is unknown or GPU variant requested but unavailable
     */
    std::shared_ptr<DeconvolutionAlgorithm> create(const DeconvolutionConfig& config);

    /**
     * @brief Get list of all available algorithms.
     * @return Vector of algorithm names
     */
    std::vector<std::string> getAvailableAlgorithms() const;

    /**
     * @brief Check if an algorithm is available.
     * @param name Algorithm name to check
     * @return True if algorithm is registered, false otherwise
     */
    bool isAlgorithmAvailable(const std::string& name) const;

private:
    DeconvolutionAlgorithmFactory() = default;
    ~DeconvolutionAlgorithmFactory() = default;
    
    // Prevent copying
    DeconvolutionAlgorithmFactory(const DeconvolutionAlgorithmFactory&) = delete;
    DeconvolutionAlgorithmFactory& operator=(const DeconvolutionAlgorithmFactory&) = delete;

    /**
     * @brief Register all available algorithms.
     * Called automatically on first access to getInstance().
     */
    void registerAlgorithms();

    bool initialized_ = false;
    std::unordered_map<std::string, AlgorithmCreator> algorithms_;
};