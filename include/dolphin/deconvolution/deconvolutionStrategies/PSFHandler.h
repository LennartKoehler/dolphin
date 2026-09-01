#pragma once
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"
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
        const DeconvolutionConfig& deconvConfig,
        const CuboidShape& maxSize);

    Result<CuboidShape> getMaxShape(
        const SetupConfig& setupConfig,
        const DeconvolutionConfig& deconvConfig);

    std::vector<std::shared_ptr<PSF>> createPSFs(
        const CuboidShape& psfShape);

private:
    CuboidShape getPSFPadding(const PSF& psf, PaddingStrategyType paddingStrategy, float paddingRelativeMax) const;

    void loadConfigsFromSetup(const SetupConfig& setupConfig);

    std::shared_ptr<ThreadPool> threadpool;
    progressCallbackFn progressFn;
    std::vector<std::shared_ptr<PSF>> psfs;
    bool psfsGenerated = false;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;
    std::vector<std::shared_ptr<PSFConfig>> inlinePsfConfigs;
    bool configsLoaded = false;
};

