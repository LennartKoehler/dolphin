#pragma once
#include "ComputationalPlan.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "../DeconvolutionConfig.h"
#include <memory>

// Forward declaration
class SetupConfig;

// creates a plan for deconvolution, this is the main logic of how to split the image, which psf is applied where etc.
// does not do any heavy compute, only creates the plan, executor the executes
class IDeconvolutionStrategy {
public:
    virtual ~IDeconvolutionStrategy() = default;
    
    // Configure the strategy with setup configuration
    virtual void configure(const SetupConfig& setupConfig) = 0;
    
    // Creates a computational plan for deconvolution
    virtual ChannelPlan createPlan(
        std::shared_ptr<ImageReader> reader,
        std::shared_ptr<ImageWriter> writer, 
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) = 0;
    

};