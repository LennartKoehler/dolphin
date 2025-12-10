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

#include "deconvolution/DeconvolutionAlgorithmFactory.h"

// Include algorithm headers
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "deconvolution/DeconvolutionConfig.h"

#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/TestAlgorithm.h"

#include <stdexcept>
#include <iostream>

DeconvolutionAlgorithmFactory& DeconvolutionAlgorithmFactory::getInstance() {
    static DeconvolutionAlgorithmFactory instance;
    if (!instance.initialized_) {
        instance.registerAlgorithms();
        instance.initialized_ = true;
    }
    return instance;
}

void DeconvolutionAlgorithmFactory::registerAlgorithm(const std::string& name, AlgorithmCreator creator) {
    if (algorithms_.find(name) != algorithms_.end()) {
        std::cerr << "[WARNING] Algorithm '" << name << "' is already registered. Overwriting." << std::endl;
    }
    algorithms_[name] = std::move(creator);
}

std::shared_ptr<DeconvolutionAlgorithm> DeconvolutionAlgorithmFactory::create(
    const DeconvolutionConfig& config
) {
    return createShared(config);
}

std::shared_ptr<DeconvolutionAlgorithm> DeconvolutionAlgorithmFactory::createShared(
    const DeconvolutionConfig& config
) {
    auto it = algorithms_.find(config.algorithmName);
    if (it == algorithms_.end()) {
        throw std::runtime_error("Unknown algorithm: " + config.algorithmName);
    }
    
    auto algorithm = it->second();
    
    // Configure the algorithm with the provided config
    algorithm->configure(config);
    return std::shared_ptr<DeconvolutionAlgorithm>(algorithm);
}

std::unique_ptr<DeconvolutionAlgorithm> DeconvolutionAlgorithmFactory::createUnique(
    const DeconvolutionConfig& config
) {
    auto it = algorithms_.find(config.algorithmName);
    if (it == algorithms_.end()) {
        throw std::runtime_error("Unknown algorithm: " + config.algorithmName);
    }
    
    auto algorithm = it->second();
    
    // Configure the algorithm with the provided config
    algorithm->configure(config);
    return std::unique_ptr<DeconvolutionAlgorithm>(algorithm);
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
    std::cout << "[INFO] Registering deconvolution algorithms..." << std::endl;
    
    registerAlgorithm("RichardsonLucy", []() {
        return new RLDeconvolutionAlgorithm();
    });
    
    registerAlgorithm("InverseFilter", []() {
        return new InverseFilterDeconvolutionAlgorithm();
    });

    registerAlgorithm("RichardsonLucyTotalVariation", []() {
        return new RLTVDeconvolutionAlgorithm();
    });
    
    registerAlgorithm("RegularizedInverseFilter", []() {
        return new RegularizedInverseFilterDeconvolutionAlgorithm();
    });
    
    registerAlgorithm("RichardsonLucywithAdaptiveDamping", []() {
        return new RLADDeconvolutionAlgorithm();
    });

    registerAlgorithm("TestAlgorithm", []() {
        return new TestAlgorithm();
    });

    std::cout << "[INFO] Registered " << algorithms_.size() << " algorithm(s)" << std::endl;
}