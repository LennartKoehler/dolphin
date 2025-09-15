#pragma once

#include <memory>
#include <utility>

// #include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"


/**
 * @brief Factory class for creating deconvolution algorithm instances with CPU/GPU variants.
 * 
 * This factory supports both CPU and GPU variants of algorithms, with conditional 
 * compilation to ensure GPU variants are only registered when CUDA is available.
 */
class DeconvolutionAlgorithmFactory {
public:
    using AlgorithmCreator = std::function<std::shared_ptr<DeconvolutionAlgorithm>()>;

    static DeconvolutionAlgorithmFactory& getInstance() {
        static DeconvolutionAlgorithmFactory instance;
        return instance;
    }

    void registerAlgorithm(const std::string& name, AlgorithmCreator creator) {
        algorithms_[name] = creator;
    }

    /**
     * @brief Create an algorithm instance based on configuration.
     * @param config Deconvolution configuration containing algorithm selection
     * @return Shared pointer to the created algorithm instance
     * @throws std::runtime_error if algorithm is unknown or GPU variant requested but unavailable
     */
    std::shared_ptr<DeconvolutionAlgorithm> create(
        const DeconvolutionConfig& config
    ) {
        auto it = algorithms_.find(config.algorithmName);
        if (it == algorithms_.end()) {
            throw std::runtime_error("Unknown algorithm: " + config.algorithmName);
        }
        
        
        auto algorithm = it->second();
        algorithm->configure(config);
        return algorithm;
    }

    /**
     * @brief Get list of all available algorithms.
     * @return Vector of algorithm names
     */
    std::vector<std::string> getAvailableAlgorithms() const {
        std::vector<std::string> names;
        for (const auto& pair : algorithms_) {
            names.push_back(pair.first);
        }
        return names;
    }



private:
    DeconvolutionAlgorithmFactory() = default;



    void registerAlgorithms() {
        // registerAlgorithm("InverseFilter", []() {
        //     return std::make_unique<InverseFilterDeconvolutionAlgorithm>();
        // });

        // registerAlgorithm("RichardsonLucyTotalVariation", []() {
        //     return std::make_unique<RLTVDeconvolutionAlgorithm>();
        // });
        // registerAlgorithm("RegularizedInverseFilter", []() {
        //     return std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
        // });
        // registerAlgorithm("RichardsonLucywithAdaptiveDamping", []() {
        //     return std::make_unique<RLADDeconvolutionAlgorithm>();
        // });
        registerAlgorithm("RichardsonLucy", []() {
            return std::make_unique<RLDeconvolutionAlgorithm>();
        });
    }


    std::unordered_map<std::string, AlgorithmCreator> algorithms_;
};
