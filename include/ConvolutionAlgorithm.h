#pragma once
#include "BaseAlgorithm.h"

class ConvolutionAlgorithm : public BaseAlgorithm{

    Hyperstack run(Hyperstack& data, std::vector<PSF>& psfs) override;
};