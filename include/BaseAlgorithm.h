#pragma once
#include "HyperstackImage.h"


class BaseAlgorithm{
public:
    BaseAlgorithm(){}
    virtual ~BaseAlgorithm() = default;
    
    virtual Hyperstack run(Hyperstack& data, std::vector<PSF>& psfs) = 0;



private:
};