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
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

#include "dolphin/psf/generators/GaussianPSFGenerator.h"
#include "dolphin/psf/generators/GibsonLanniModel.h"
#include "dolphin/psf/generators/PSFGenerator.h"
#include "dolphin/psf/modality/ModalityStrategy.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/ThreadPool.h"

#include <spdlog/spdlog.h>


// Factory singleton which creates PSFConfigs and minimal PSF generators,
// keyed by the model name. The factory never generates a PSF itself: it
// assembles a generator that already holds everything needed to generate - a
// modality strategy (the orchestrator) and a PSF model, both derived from the
// config, plus the config. The caller provides the thread pool via
// configure() and runs generatePSF().
class PSFGeneratorFactory {
public:
    using ConfigCreator = std::function<std::shared_ptr<PSFConfig>()>;
    using GeneratorCreator = std::function<std::shared_ptr<BasePSFGenerator>(std::shared_ptr<PSFConfig>)>;

    static PSFGeneratorFactory& getInstance() {
        static PSFGeneratorFactory instance;
        return instance;
    }

    void registerPSFType(const std::string& name,
                         ConfigCreator configCreator,
                         GeneratorCreator generatorCreator) {
        generators_[name] = std::move(generatorCreator);
        configs_[name] = std::move(configCreator);
    }

    /** Loads a PSF config from JSON (model selection + modality selection). */
    std::shared_ptr<PSFConfig> createConfig(const json& configJson) {
        std::string psfModel;
        if (!configJson.is_object()) {
            throw std::runtime_error(
                "PSF config must be a JSON object with 'model_name'. "
                "Found type: " + std::string(configJson.type_name()) + " (value: " + configJson.dump() + ")"
            );
        }
        try{
            psfModel = configJson.at("model_name").get<std::string>();
        }
        catch (...) {
            throw std::runtime_error(
                "'model_name' field missing from PSF config. "
                "Expected a JSON object containing 'model_name' (e.g. \"Gaussian\" or \"GibsonLanni\"). "
                "Config was: " + configJson.dump()
            );
        }


        auto it = configs_.find(psfModel);
        if (it == configs_.end()) {
            throw std::runtime_error("Unknown PSF model: " + psfModel);
        }

        auto config = it->second();
        if (!config->loadFromJSON(configJson)) {
            throw std::runtime_error("Failed to load PSF config for: " + psfModel);
        }

        config->printValues();
        // Generic sanity pass over every registered parameter (base +
        // modality + model). Warnings only: each generator validates the
        // values it actually depends on with more context.
        config->logParameterRangeViolations();
        return config;
    }

    /** Assembles a generator for an already-loaded config: the modality
     *  strategy and PSF model are derived from it here, once. */
    std::shared_ptr<BasePSFGenerator> createGenerator(std::shared_ptr<PSFConfig> config) {
        if (!config)
            throw std::runtime_error("PSFGeneratorFactory: no PSF config given");

        auto it = generators_.find(config->getModelName());
        if (it == generators_.end()) {
            throw std::runtime_error("Unknown PSF model: " + config->getModelName());
        }
        return it->second(std::move(config));
    }

    /** Assembles a generator from a JSON config (see createConfig). */
    std::shared_ptr<BasePSFGenerator> createGenerator(const json& configJson) {
        return createGenerator(createConfig(configJson));
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
            []() { return std::make_shared<GaussianPSFConfig>(); },
            [](std::shared_ptr<PSFConfig> config) {
                // The analytic Gaussian model is self-contained: no modality
                // layer, the generator drives the model directly. A modality
                // tag other than widefield would be silently ignored, so say so.
                if (config->getModalityType() != MicroscopeModality::WIDEFIELD)
                    spdlog::warn("Gaussian PSF model has no '{}' image formation; ignoring microscope_modality (widefield PSF will be generated)",
                                 config->getModalityName());
                auto generator = std::make_shared<GaussianPSFGenerator>();
                generator->setConfig(std::move(config));
                return generator;
            }
        );

        registerPSFType("GibsonLanni",
            []() { return std::make_shared<GibsonLanniPSFConfig>(); },
            [](std::shared_ptr<PSFConfig> config) {
                auto* glConfig = dynamic_cast<GibsonLanniPSFConfig*>(config.get());
                if (!glConfig)
                    throw std::runtime_error("Wrong config type for GibsonLanni model");
                return std::make_shared<PSFGenerator>(std::move(config),
                                                      ModalityStrategy::forConfig(*glConfig),
                                                      std::make_shared<GibsonLanniModel>());
            }
        );
    }

    std::unordered_map<std::string, GeneratorCreator> generators_;
    std::unordered_map<std::string, ConfigCreator> configs_;
};
