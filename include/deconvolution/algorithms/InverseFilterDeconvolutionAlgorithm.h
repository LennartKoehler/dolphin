#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "psf/PSF.h"
#include <iostream>
#include <vector>
#ifdef CUDA_AVAILABLE
#include <cufft.h>
#include <cufftw.h>
#else
#include <fftw3.h>
#endif

class InverseFilterDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm{
public:
    void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    void configure(const DeconvolutionConfig& config) override;

private:
};

