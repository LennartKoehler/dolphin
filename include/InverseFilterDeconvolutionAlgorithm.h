#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "PSF.h"
#include <iostream>
#include <vector>
#include <fftw3.h>

class InverseFilterDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm{
public:
    Hyperstack deconvolve(Hyperstack& data, std::vector<PSF>& psf) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    double epsilon;
};

