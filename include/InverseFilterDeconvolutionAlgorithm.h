#pragma once

#include "BaseDeconvolutionAlgorithm.h"
#include "PSF.h"
#include <iostream>
#include <vector>
#include <fftw3.h>

class InverseFilterDeconvolutionAlgorithm : public BaseDeconvolutionAlgorithm{
public:
    void algorithm(Hyperstack& data, int channel_num) override;
    void configure(const DeconvolutionConfig& config) override;

private:
};

