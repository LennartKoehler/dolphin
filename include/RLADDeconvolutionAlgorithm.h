#pragma once

#include "BaseDeconvolutionAlgorithm.h"
// #include "PSF.h"
#include <iostream>


class RLADDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm {
public:
    void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    int iterations;
    int dampingDecrease; //0=exp, 1=lin
    double alpha;
    double beta;

};