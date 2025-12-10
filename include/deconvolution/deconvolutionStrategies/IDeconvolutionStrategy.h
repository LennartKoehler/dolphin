#pragma once
#include "ComputationalPlan.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "DeconvolutionConfig.h"
#include <memory>


// creates a plan for deconvolution, this is the main logic of how to split the image, which psf is applied where etc.
// does not do any heavy compute, only creates the plan, executor the executes
class IDeconvolutionStrategy {
public:
    virtual ~IDeconvolutionStrategy() = default;
    
    // Creates a computational plan for deconvolution
    virtual ChannelPlan createPlan(
        const ImageMetaData& metadata,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config) = 0;
    

};