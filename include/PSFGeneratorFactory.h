#pragma once

#include "GaussianPSFGenerator.h"
#include "GibsonLanniPSFGenerator.h"

namespace PSFFactory{

    static std::unique_ptr<BasePSFGenerator> PSFGeneratorFactory(const std::string& psfModelName, const json& jsonData){
        std::unique_ptr<BasePSFGenerator> psfGenerator;
        std::unique_ptr<PSFConfig> psfConfig;

        if (psfModelName == "Gaussian"){
            psfConfig = std::make_unique<GaussianPSFConfig>();
            psfGenerator = std::make_unique<GaussianPSFGenerator>();
        }
        else if (psfModelName == "GibsonLanni"){
            psfConfig = std::make_unique<GibsonLanniPSFConfig>();
            psfGenerator = std::make_unique<GibsonLanniPSFGenerator>();
        }
        assert(psfGenerator != nullptr && "could not create psf generator given the model name");

        psfGenerator->setConfig(std::move(psfConfig));
        return psfGenerator;
    }

    static std::unique_ptr<BasePSFGenerator> PSFGeneratorFactory(std::unique_ptr<PSFConfig> config){
        std::unique_ptr<BasePSFGenerator> psfGenerator;
        if (config->getName() == "Gaussian"){
            psfGenerator = std::make_unique<GaussianPSFGenerator>();
        }
        else if (config->getName() == "GibsonLanni"){
            psfGenerator = std::make_unique<GibsonLanniPSFGenerator>();
        }
        assert(psfGenerator != nullptr && "could not create psf generator given the model name");

        psfGenerator->setConfig(std::move(config));
        return psfGenerator;

    }

    static std::unique_ptr<PSFConfig> PSFConfigFactory(const json& configJson) {

        std::string psfModel = configJson["psfmodel"].get<std::string>();
        
        std::unique_ptr<PSFConfig> config;
        if (psfModel == "Gaussian") {
            config = std::make_unique<GaussianPSFConfig>();
        } else if (psfModel == "GibsonLanni") {
            config = std::make_unique<GibsonLanniPSFConfig>();
        } else {
            throw std::runtime_error("Unknown PSF model: " + psfModel);
        }
        
        if (!config->loadFromJSON(configJson)) {
            throw std::runtime_error("Failed to load PSF config");
        }
        
        config->printValues();
        return config;
    }

}