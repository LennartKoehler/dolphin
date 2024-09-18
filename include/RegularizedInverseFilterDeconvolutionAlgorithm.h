#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "HyperstackImage.h"
#include "PSF.h"
#include <iostream>

class RegularizedInverseFilterDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm {
public:
    Hyperstack deconvolve(Hyperstack& data, PSF& psf) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    double epsilon;
    bool grid;
    double lambda;
    int borderType;
    int psfSafetyBorder;
    int cubeSize;
};
