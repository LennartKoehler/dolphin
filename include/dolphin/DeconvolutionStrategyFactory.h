#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include "deconvolution/deconvolutionStrategies/IDeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/IDeconvolutionExecutor.h"
#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategyPair.h"

// Forward declarations
class SetupConfig;

/**
 * Factory for creating deconvolution strategy pairs based on configuration type.
 * Each strategy pair contains both a strategy and its corresponding executor.
 * Supports both built-in strategies and custom strategy registration.
 */
class DeconvolutionStrategyFactory {
public:
    using StrategyPairCreator = std::function<std::unique_ptr<DeconvolutionStrategyPair>(std::shared_ptr<SetupConfig>)>;
    
    /**
     * Get the singleton instance of the factory
     */
    static DeconvolutionStrategyFactory& getInstance();
    
    /**
     * Create a deconvolution strategy pair based on the type string
     * @param type The deconvolution type string (e.g., "normal", "labeled")
     * @return Unique pointer to the created strategy pair, or nullptr if type is unknown
     */
    std::unique_ptr<DeconvolutionStrategyPair> createStrategyPair(const std::string& type, std::shared_ptr<SetupConfig> config);

    /**
     * Create a deconvolution strategy pair based on SetupConfig
     * Automatically determines strategy type and loads required data (labeled images, PSF maps)
     * @param setupConfig The setup configuration containing all necessary information
     * @return Unique pointer to the configured strategy pair, or nullptr if configuration is invalid
     */
    std::unique_ptr<DeconvolutionStrategyPair> createStrategyPair(std::shared_ptr<SetupConfig> setupConfig);

    /**
     * Create a deconvolution strategy based on SetupConfig (backward compatibility)
     * Automatically determines strategy type and loads required data (labeled images, PSF maps)
     * @param setupConfig The setup configuration containing all necessary information
     * @return Unique pointer to the configured strategy, or nullptr if configuration is invalid
     */
    std::unique_ptr<IDeconvolutionStrategy> createStrategy(std::shared_ptr<SetupConfig> setupConfig);
    
    /**
     * Register a custom strategy pair creator
     * @param type The type string to associate with this strategy
     * @param creator Function that creates and returns the strategy pair
     */
    void registerStrategy(const std::string& type, StrategyPairCreator creator);
    
    /**
     * Check if a strategy type is supported
     * @param type The type string to check
     * @return true if the type is registered, false otherwise
     */
    bool isStrategySupported(const std::string& type) const;
    
    /**
     * Get list of all registered strategy types
     * @return Vector of registered type strings
     */
    std::vector<std::string> getSupportedTypes() const;

private:
    DeconvolutionStrategyFactory();
    ~DeconvolutionStrategyFactory() = default;
    
    // Delete copy constructor and assignment operator
    DeconvolutionStrategyFactory(const DeconvolutionStrategyFactory&) = delete;
    DeconvolutionStrategyFactory& operator=(const DeconvolutionStrategyFactory&) = delete;
    
    /**
     * Register built-in strategies
     */
    void registerBuiltInStrategies();
    
    std::unordered_map<std::string, StrategyPairCreator> strategy_creators_;
};
