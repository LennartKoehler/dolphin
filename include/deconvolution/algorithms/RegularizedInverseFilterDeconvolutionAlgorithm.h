#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "HyperstackImage.h"
// #include "PSF.h"
#include <iostream>

class RegularizedInverseFilterDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm {
public:
    void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    double lambda;

};
