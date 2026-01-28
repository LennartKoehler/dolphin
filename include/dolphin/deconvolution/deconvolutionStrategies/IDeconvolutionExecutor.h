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

class ImageReader;
class ImageWriter;
/*
This class runs the DeconvolutionPlan provided by the deconvolutionstrategy
*/

class IDeconvolutionExecutor {
public:
    virtual ~IDeconvolutionExecutor() = default;
    
    // Execute a computational plan and return the result
    virtual void execute(const DeconvolutionPlan& plan) = 0;
    
    // Configure the executor with necessary parameters
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) = 0;
};