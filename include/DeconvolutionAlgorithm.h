#pragma once

#include <memory>
#include <utility>
#include "BaseDeconvolutionAlgorithm.h"
#include "DeconvolutionConfig.h"

template<typename Algorithm, typename... Args>
class DeconvolutionAlgorithm {
public:
    DeconvolutionAlgorithm(const DeconvolutionConfig& config, Args&&... args)
            : algo(std::make_unique<Algorithm>(std::forward<Args>(args)...)) {
        algo->configure(config);
    }

    Hyperstack deconvolve(Hyperstack& data, PSF& psf) const {
        return algo->deconvolve(data, psf);
    }

private:
    std::unique_ptr<BaseDeconvolutionAlgorithm> algo;
};
