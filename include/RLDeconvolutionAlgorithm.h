#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "PSF.h"
#include <iostream>
#include <fftw3.h>

class RLDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm {
public:
    Hyperstack deconvolve(Hyperstack& data, PSF& psf) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    int iterations;
    double epsilon;
};
