#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "PSF.h"
#include <iostream>
#include <fftw3.h>

class RLTVDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm {
public:
    void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    int iterations;
    double lambda;
};