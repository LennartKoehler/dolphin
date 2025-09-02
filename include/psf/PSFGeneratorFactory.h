#pragma once
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

#include "psf/generators/GaussianPSFGenerator.h"
#include "psf/generators/GibsonLanniPSFGenerator.h"
#include "psf/configs/GibsonLanniPSFConfig.h"
#include "psf/configs/GaussianPSFConfig.h"
#include "psf/generators/BasePSFGenerator.h"
#include "psf/configs/PSFConfig.h"


// factory singleton which creates either PSFConfigs or PSFGenerators, usually using the string name
class PSFGeneratorFactory {
public:
    using GeneratorCreator = std::function<std::shared_ptr<BasePSFGenerator>()>;
    using ConfigCreator = std::function<std::shared_ptr<PSFConfig>()>;

    static PSFGeneratorFactory& getInstance() {
        static PSFGeneratorFactory instance;
        return instance;
    }

    void registerPSFType(const std::string& name, 
                         GeneratorCreator generatorCreator,
                         ConfigCreator configCreator) {
        generators_[name] = generatorCreator;
        configs_[name] = configCreator;
    }

    std::shared_ptr<BasePSFGenerator> createGenerator(
        const std::string& name, 
        const json& configData
    ) {
        auto genIt = generators_.find(name);
        auto confIt = configs_.find(name);
        
        if (genIt == generators_.end() || confIt == configs_.end()) {
            throw std::runtime_error("Unknown PSF model: " + name);
        }
        
        // Create config and load data
        auto config = confIt->second();
        if (!config->loadFromJSON(configData)) {
            throw std::runtime_error("Failed to load PSF config for: " + name);
        }
        config->printValues();
        
        // Create generator and set config
        auto generator = genIt->second();
        generator->setConfig(std::move(config));
        
        return generator;
    }

    std::shared_ptr<BasePSFGenerator> createGenerator(std::shared_ptr<PSFConfig> config) {
        std::string modelName = config->getName();
        
        auto it = generators_.find(modelName);
        if (it == generators_.end()) {
            throw std::runtime_error("Unknown PSF model: " + modelName);
        }
        
        auto generator = it->second();
        generator->setConfig(std::move(config));
        return generator;
    }

    std::shared_ptr<PSFConfig> createConfig(const json& configJson) {
        std::string psfModel = configJson["psfModel"].get<std::string>();
        
        auto it = configs_.find(psfModel);
        if (it == configs_.end()) {
            throw std::runtime_error("Unknown PSF model: " + psfModel);
        }
        
        auto config = it->second();
        if (!config->loadFromJSON(configJson)) {
            throw std::runtime_error("Failed to load PSF config for: " + psfModel);
        }
    
        config->printValues();
        return config;
    }

    std::vector<std::string> getAvailablePSFModels() const {
        std::vector<std::string> names;
        for (const auto& pair : generators_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    PSFGeneratorFactory() {
        // Register PSF types
        registerPSFType("Gaussian", 
            []() { return std::make_unique<GaussianPSFGenerator>(); },
            []() { return std::make_unique<GaussianPSFConfig>(); }
        );
        
        registerPSFType("GibsonLanni",
            []() { return std::make_unique<GibsonLanniPSFGenerator>(); },
            []() { return std::make_unique<GibsonLanniPSFConfig>(); }
        );
    }

    std::unordered_map<std::string, GeneratorCreator> generators_;
    std::unordered_map<std::string, ConfigCreator> configs_;
};

// Backwards compatibility namespace (optional)
namespace PSFFactory {
    inline std::shared_ptr<BasePSFGenerator> PSFGeneratorFactory(
        const std::string& psfModelName, 
        const json& jsonData
    ) {
        return ::PSFGeneratorFactory::getInstance().createGenerator(psfModelName, jsonData);
    }

    inline std::shared_ptr<BasePSFGenerator> PSFGeneratorFactory(
        std::shared_ptr<PSFConfig> config
    ) {
        return ::PSFGeneratorFactory::getInstance().createGenerator(std::move(config));
    }

    inline std::shared_ptr<PSFConfig> PSFConfigFactory(const json& configJson) {
        return ::PSFGeneratorFactory::getInstance().createConfig(configJson);
    }
}