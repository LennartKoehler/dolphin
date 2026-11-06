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
#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/ProgressTracking.h"
#include "dolphin/ServiceAbstractions.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/MemoryComponent.h"
#include <memory>


/**
 * @brief MemoryComponent for the preprocessed PSF cache.
 *
 * The cache itself lives on the PSFPreprocessor instances (one per device,
 * created by the handler), but the handler is the access point: it knows how
 * many PSFs exist and is what the strategy already holds. Reports one padded
 * complex PSF per PSF per device held simultaneously in the caches.
 */
class PSFCacheMemoryComponent : public MemoryComponent {
public:
    PSFCacheMemoryComponent() = default;

    // Configured by the strategy: one cached entry per PSF per device.
    void configure(size_t numPSFs, size_t numDevices) {
        nPSFs = numPSFs;
        nDevices = numDevices;
    }

    MemoryRequirement footprint(const CuboidShape& cubeShape) const override {
        // Padded complex PSFs: 2 x real_t per element.
        const size_t bytesPerPsf = cubeShape.getVolume() * sizeof(complex_t);
        return MemoryRequirement{0, nPSFs * nDevices * bytesPerPsf};
    }

    std::string name() const override { return "psf cache"; }

private:
    size_t nPSFs = 0;
    size_t nDevices = 1;
};


class PSFHandler{
public:

    PSFHandler(std::shared_ptr<ThreadPool> threadpool, progressCallbackFn fn) : threadpool(threadpool), progressFn(fn){}
    std::unique_ptr<PSFPreprocessor> createPSFPreprocessor() const ;

    void generatePSFs(
        const SetupConfig& setupConfig,
        const DeconvolutionConfig& deconvConfig,
        const CuboidShape& maxSize);

    Result<PaddingScheme> getPadding(
        const DeconvolutionConfig& deconvConfig) const;

    CuboidShape getMaxShape() const;

    std::vector<std::shared_ptr<PSF>>& getPSFs(const CuboidShape& shape);

    size_t getPSFCount() const { return psfs.size(); }

    // MemoryComponent of the preprocessed PSF cache, configured for numPSFs
    // cached PSFs on numDevices devices. Owned by the handler.
    MemoryComponent& getMemoryComponent(size_t numPSFs, size_t numDevices) {
        psfCacheComponent.configure(numPSFs, numDevices);
        return psfCacheComponent;
    }

private:
    static std::string paddingStrategyName(PaddingStrategyType type);
    static float generationCutoffThreshold(const DeconvolutionConfig& deconvConfig);
    static CuboidShape getManualCubePadding(const std::array<int64_t, 3>& cubePadding);
    CuboidShape getManualRelativeMaxPadding(const PSF& psf, float paddingRelativeMax) const;
    CuboidShape getPresetPadding(const PSF& psf, PaddingStrategyType paddingType) const;
    Result<PaddingScheme> makePaddingScheme(CuboidShape totalPadding) const;

    void loadConfigsFromSetup(const SetupConfig& setupConfig);

    std::shared_ptr<ThreadPool> threadpool;
    progressCallbackFn progressFn;
    std::vector<std::shared_ptr<PSF>> psfs;
    bool psfsGenerated = false;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;
    bool configsLoaded = false;
    PSFCacheMemoryComponent psfCacheComponent;
};

