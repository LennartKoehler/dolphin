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
#include "dolphin/deconvolution/deconvolutionStrategies/IDeconvolutionExecutor.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"

class SetupConfig;

class StandardDeconvolutionExecutor : public IDeconvolutionExecutor {
public:
    StandardDeconvolutionExecutor();
    virtual ~StandardDeconvolutionExecutor();

    // IDeconvolutionExecutor interface
    virtual void execute(const DeconvolutionPlan& plan) override;
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) override;
    virtual void configure(const SetupConfig& setupConfig);

protected:
    // Helper methods for execution
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc);
    
    virtual void runTask(const CubeTaskDescriptor& task);
    // Parallel execution
    virtual void parallelDeconvolution(
        const DeconvolutionPlan& channelPlan);

protected:
    LoadingBar loadingBar;
 };