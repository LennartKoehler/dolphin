#include "deconvolution/DeconvolutionAlgorithmFactory.h"

// Include algorithm headers
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/DeconvolutionConfig.h"

// Uncomment these as you add more algorithms
// #include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"

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
    auto it = algorithms_.find(config.algorithmName);
    if (it == algorithms_.end()) {
        throw std::runtime_error("Unknown algorithm: " + config.algorithmName);
    }
    
    auto algorithm = it->second();
    
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
    std::cout << "[INFO] Registering deconvolution algorithms..." << std::endl;
    
    // Register Richardson-Lucy algorithm
    registerAlgorithm("RichardsonLucy", []() {
        return std::make_shared<RLDeconvolutionAlgorithm>();
    });
    
    // Register other algorithms as they become available
    // registerAlgorithm("InverseFilter", []() {
    //     return std::make_shared<InverseFilterDeconvolutionAlgorithm>();
    // });

    // registerAlgorithm("RichardsonLucyTotalVariation", []() {
    //     return std::make_shared<RLTVDeconvolutionAlgorithm>();
    // });
    
    // registerAlgorithm("RegularizedInverseFilter", []() {
    //     return std::make_shared<RegularizedInverseFilterDeconvolutionAlgorithm>();
    // });
    
    // registerAlgorithm("RichardsonLucywithAdaptiveDamping", []() {
    //     return std::make_shared<RLADDeconvolutionAlgorithm>();
    // });

    std::cout << "[INFO] Registered " << algorithms_.size() << " algorithm(s)" << std::endl;
}