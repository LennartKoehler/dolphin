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

#include "DeconvolutionStrategyFactory.h"
#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.h"
#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "frontend/SetupConfig.h"
#include "HyperstackImage.h"
#include <stdexcept>

DeconvolutionStrategyFactory::DeconvolutionStrategyFactory() {
    registerBuiltInStrategies();
}

DeconvolutionStrategyFactory& DeconvolutionStrategyFactory::getInstance() {
    static DeconvolutionStrategyFactory instance;
    return instance;
}

std::unique_ptr<DeconvolutionStrategyPair> DeconvolutionStrategyFactory::createStrategyPair(std::shared_ptr<SetupConfig> config) {
    std::string type = config->strategyType;
    auto it = strategy_creators_.find(type);
    if (it != strategy_creators_.end()) {
        return it->second(config);
    }
    
    // Return nullptr for unknown types
    return nullptr;
}

std::unique_ptr<IDeconvolutionStrategy> DeconvolutionStrategyFactory::createStrategy(std::shared_ptr<SetupConfig> config) {
    std::string type = config->strategyType;
    auto it = strategy_creators_.find(type);
    if (it != strategy_creators_.end()) {
        // Extract just the strategy from the pair
        auto pair = it->second(config);
        if (pair) {
            // Create a copy of the strategy by cloning it
            // This is a simplified approach - in a real implementation,
            // strategies should have proper clone methods
            return std::make_unique<StandardDeconvolutionStrategy>(); // Placeholder
        }
    }
    
    // Return nullptr for unknown types
    return nullptr;
}

void DeconvolutionStrategyFactory::registerStrategy(const std::string& type, StrategyPairCreator creator) {
    if (!creator) {
        throw std::invalid_argument("Strategy creator cannot be null");
    }
    strategy_creators_[type] = creator;
}

bool DeconvolutionStrategyFactory::isStrategySupported(const std::string& type) const {
    return strategy_creators_.find(type) != strategy_creators_.end();
}

std::vector<std::string> DeconvolutionStrategyFactory::getSupportedTypes() const {
    std::vector<std::string> types;
    types.reserve(strategy_creators_.size());
    
    for (const auto& pair : strategy_creators_) {
        types.push_back(pair.first);
    }
    
    return types;
}

void DeconvolutionStrategyFactory::registerBuiltInStrategies() {
    registerStrategy("normal", [](std::shared_ptr<SetupConfig> setupConfig) -> std::unique_ptr<DeconvolutionStrategyPair> {
        auto strategy = std::make_unique<StandardDeconvolutionStrategy>();
        auto executor = std::make_unique<StandardDeconvolutionExecutor>();
        
        // Configure strategy with setup config
        strategy->configure(*setupConfig);
        executor->configure(*setupConfig);
        
        return std::make_unique<DeconvolutionStrategyPair>(std::move(strategy), std::move(executor));
    });
    
    // Register labeled image deconvolution strategy
    registerStrategy("labeled", [](std::shared_ptr<SetupConfig> setupConfig) -> std::unique_ptr<DeconvolutionStrategyPair> {
        auto strategy = std::make_unique<StandardDeconvolutionStrategy>();
        auto executor = std::make_unique<LabeledDeconvolutionExecutor>();

        // Configure strategy with setup config (loads labeled image and PSF map internally)
        strategy->configure(*setupConfig);
        executor->configure(*setupConfig);

        return std::make_unique<DeconvolutionStrategyPair>(std::move(strategy), std::move(executor));
    });
}
