#pragma once

#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "PSF.h"



class BaseDeconvolutionAlgorithm {
public:
    virtual ~BaseDeconvolutionAlgorithm() = default;
    virtual Hyperstack deconvolve(Hyperstack& data, PSF& psf) = 0;
    virtual void configure(const DeconvolutionConfig& config) = 0;
};
