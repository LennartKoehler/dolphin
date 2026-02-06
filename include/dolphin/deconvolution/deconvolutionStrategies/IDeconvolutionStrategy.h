/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include <memory>

#include "dolphin/ServiceAbstractions.h"
class SetupConfig;

/*
IDeconvolutionStrategy creates a plan for the deconvolution of an image. The IDeconvolutionExecutor then uses this plan to execute the deconvolution.
The plan consists of many TaskDescriptions. These tell the executor what part of the image should be processed and how.
The DeconvolutionStrategy also handles how the backends are supposed to be used by adding a Context to the TaskDescription.
*/
class IDeconvolutionStrategy {
public:
    virtual ~IDeconvolutionStrategy() = default;
    
    // Configure the strategy with setup configuration
    virtual void configure(const SetupConfig& setupConfig) = 0;
    
    // Creates a computational plan for deconvolution
    virtual Result<DeconvolutionPlan> createPlan(
        std::shared_ptr<ImageReader> reader,
        std::shared_ptr<ImageWriter> writer, 
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) = 0;
    

};