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

    void generatePSFs(
        const SetupConfig& setupConfig,
        const CuboidShape& maxSize);

    Result<PaddingScheme> getPadding(
        const DeconvolutionConfig& deconvConfig) const;

    Result<CuboidShape> getMaxShape() const;

    const std::vector<std::shared_ptr<PSF>>& getPSFs() const { return psfs; }

    void fitPSFsToShape(const CuboidShape& targetShape);

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

