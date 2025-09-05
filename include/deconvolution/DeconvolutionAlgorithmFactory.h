#pragma once

#include <memory>
#include <utility>

#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"


class DeconvolutionAlgorithmFactory {
public:
    using AlgorithmCreator = std::function<std::shared_ptr<BaseDeconvolutionAlgorithm>()>;

    static DeconvolutionAlgorithmFactory& getInstance() {
        static DeconvolutionAlgorithmFactory instance;
        return instance;
    }

    void registerAlgorithm(const std::string& name, AlgorithmCreator creator) {
        algorithms_[name] = creator;
    }

    std::shared_ptr<BaseDeconvolutionAlgorithm> create(
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

    std::vector<std::string> getAvailableAlgorithms() const {
        std::vector<std::string> names;
        for (const auto& pair : algorithms_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    DeconvolutionAlgorithmFactory() {
        // Register algorithms
        registerAlgorithm("InverseFilter", []() {
            return std::make_unique<InverseFilterDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucy", []() {
            return std::make_unique<RLDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucyTotalVariation", []() {
            return std::make_unique<RLTVDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RegularizedInverseFilter", []() {
            return std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucywithAdaptiveDamping", []() {
            return std::make_unique<RLADDeconvolutionAlgorithm>();
        });
    }
    std::unordered_map<std::string, AlgorithmCreator> algorithms_;
};
