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


    Result<Padding> getPadding(
        const SetupConfig& setupConfig,
        const DeconvolutionConfig& deconvConfig,
        const CuboidShape& imageSize);

    std::vector<std::shared_ptr<PSF>> createPSFs(
        const CuboidShape& psfShape);

private:
    CuboidShape getPSFPadding(const PSF& psf, PaddingStrategyType paddingStrategy, float paddingRelativeMax) const;
    CuboidShape getPaddingFromConfig(std::shared_ptr<PSFConfig> config, PaddingStrategyType paddingStrategy) const;

    std::shared_ptr<ThreadPool> threadpool;
    progressCallbackFn progressFn;
    std::vector<PSF> filePSFs;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;
};

