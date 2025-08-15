#pragma once

#include <memory>
#include <utility>
// #include "BaseDeconvolutionAlgorithm.h"
// #include "DeconvolutionConfig.h"
#include "InverseFilterDeconvolutionAlgorithm.h"
#include "RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "RLDeconvolutionAlgorithm.h"
#include "RLTVDeconvolutionAlgorithm.h"
#include "RLADDeconvolutionAlgorithm.h"

// template<typename Algorithm, typename... Args>
// class DeconvolutionAlgorithm {
// public:
//     DeconvolutionAlgorithm(const DeconvolutionConfig& config, Args&&... args)
//             : algo(std::make_unique<Algorithm>(std::forward<Args>(args)...)) {
//         algo->configure(config);
//     }

//     Hyperstack deconvolve(Hyperstack& data, std::vector<PSF>& psfs) const {
//         return algo->deconvolve(data, psfs);
//     }

// private:
//     std::unique_ptr<BaseDeconvolutionAlgorithm> algo;
// };

//TODO this entire file can be removed and this factory included somewhere else
static std::unique_ptr<BaseDeconvolutionAlgorithm> deconvolutionAlgorithmFactory(
    const std::string& name, const DeconvolutionConfig& config
) {
    std::unique_ptr<BaseDeconvolutionAlgorithm> algorithm;
    if (name == "InverseFilter") {
        algorithm = std::make_unique<InverseFilterDeconvolutionAlgorithm>();
    } else if (name == "RichardsonLucy") {
        algorithm = std::make_unique<RLDeconvolutionAlgorithm>();
    } else if (name == "RichardsonLucyTotalVariation") {
        algorithm = std::make_unique<RLTVDeconvolutionAlgorithm>();
    } else if (name == "RegularizedInverseFilter") {
        algorithm = std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
    }
    else{
        throw std::runtime_error("Unknown algorithm: " + name);
    }    
    algorithm->configure(config);
    return algorithm;
}