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
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/ProgressTracking.h"
#include "dolphin/ServiceAbstractions.h"


class PSFHandler{
public:

    PSFHandler(std::shared_ptr<ThreadPool> threadpool, progressCallbackFn fn) : threadpool(threadpool), progressFn(fn){}
    std::unique_ptr<PSFPreprocessor> createPSFPreprocessor() const ;

    void setInlinePSFConfigs(std::vector<std::shared_ptr<PSFConfig>> configs) { inlinePsfConfigs = std::move(configs); }
    bool hasInlineConfigs() const { return !inlinePsfConfigs.empty(); }


    Result<Padding> getPadding(
        const SetupConfig& setupConfig,
        const DeconvolutionConfig& deconvConfig);

    Result<CuboidShape> getMaxShape(
        const SetupConfig& setupConfig,
        const DeconvolutionConfig& deconvConfig);

    std::vector<std::shared_ptr<PSF>> createPSFs(
        const CuboidShape& psfShape);

private:
    CuboidShape getPSFPadding(const PSF& psf, PaddingStrategyType paddingStrategy, float paddingRelativeMax) const;
    CuboidShape getPaddingFromConfig(std::shared_ptr<PSFConfig> config, PaddingStrategyType paddingStrategy) const;

    void loadConfigsFromSetup(const SetupConfig& setupConfig);

    std::shared_ptr<ThreadPool> threadpool;
    progressCallbackFn progressFn;
    std::vector<PSF> filePSFs;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;
    std::vector<std::shared_ptr<PSFConfig>> inlinePsfConfigs;
    bool configsLoaded = false;
};

