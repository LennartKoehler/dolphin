#pragma once
#include "ComputationalPlan.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "DeconvolutionConfig.h"
#include <memory>


// the class which actually runs the deconvolution tasks described in channelplan
// uses reader and writer itself to manage io
class IDeconvolutionExecutor {
public:
    virtual ~IDeconvolutionExecutor() = default;
    
    // Execute a computational plan and return the result
    virtual void execute(const ChannelPlan& plan, const ImageReader& reader, const ImageWriter& writer) = 0;
    
    // Configure the executor with necessary parameters
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) = 0;
};