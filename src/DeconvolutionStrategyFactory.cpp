#include "DeconvolutionStrategyFactory.h"
#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/LabeledImageDeconvolutionStrategy.h"
#include "frontend/SetupConfig.h"
#include "HyperstackImage.h"
#include "UtlIO.h"
#include <stdexcept>

DeconvolutionStrategyFactory::DeconvolutionStrategyFactory() {
    registerBuiltInStrategies();
}

DeconvolutionStrategyFactory& DeconvolutionStrategyFactory::getInstance() {
    static DeconvolutionStrategyFactory instance;
    return instance;
}

std::unique_ptr<DeconvolutionStrategy> DeconvolutionStrategyFactory::createStrategy(std::shared_ptr<SetupConfig> config) {
    std::string type = config->strategyType;
    auto it = strategy_creators_.find(type);
    if (it != strategy_creators_.end()) {
        return it->second(config);
    }
    
    // Return nullptr for unknown types
    return nullptr;
}

void DeconvolutionStrategyFactory::registerStrategy(const std::string& type, StrategyCreator creator) {
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
    registerStrategy("normal", [](std::shared_ptr<SetupConfig>) -> std::unique_ptr<DeconvolutionStrategy> {
        return std::make_unique<StandardDeconvolutionStrategy>();
    });
    
    // Register labeled image deconvolution strategy
    // TODO change, this should not actually have config reading 
    registerStrategy("labeled", [](std::shared_ptr<SetupConfig> setupConfig) -> std::unique_ptr<DeconvolutionStrategy> {
        std::unique_ptr<LabeledImageDeconvolutionStrategy> strat = std::make_unique<LabeledImageDeconvolutionStrategy>();

        std::shared_ptr<Hyperstack> labels = std::make_shared<Hyperstack>();
        labels->readFromTifFile(setupConfig->labeledImage.c_str());
        strat->setLabeledImage(labels);

        // load json
        RangeMap<std::string> labelPSFMap;
        labelPSFMap.loadFromString(setupConfig->labelPSFMap);
        strat->setLabelPSFMap(labelPSFMap);

        return strat;

    });
}
