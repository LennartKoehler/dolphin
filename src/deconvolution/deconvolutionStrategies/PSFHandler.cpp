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
#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/PSFCreator.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/DefaultParameters.h"
#include "dolphin_image/ImagePadding.h"
#include <cstdint>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <sys/types.h>



std::string PSFHandler::paddingStrategyName(PaddingStrategyType type) {
    switch(type){
        case(DEFAULT): return "DEFAULT";
        case(NONE): return "NONE";
        case(LOW): return "LOW";
        case(MEDIUM): return "MEDIUM";
        case(MAX): return "MAX";
        default: return "UNKNOWN";
    }
}

CuboidShape PSFHandler::getManualCubePadding(const std::array<int64_t, 3>& cubePadding) {
    return CuboidShape{
        static_cast<size_t>(std::max<uint64_t>(0, cubePadding[0])),
        static_cast<size_t>(std::max<uint64_t>(0, cubePadding[1])),
        static_cast<size_t>(std::max<uint64_t>(0, cubePadding[2]))
    };
}

CuboidShape PSFHandler::getManualRelativeMaxPadding(const PSF& psf, float paddingRelativeMax) const {
    return psf.getRegionLargerThreshold(paddingRelativeMax * psf.getMax());
}

CuboidShape PSFHandler::getPresetPadding(const PSF& psf, PaddingStrategyType paddingType) const {
    const auto& defaults = DefaultParameters::getDefaultParameters();
    switch(paddingType){
        case(DEFAULT):
            [[fallthrough]];
        case(MEDIUM):
            return psf.getRegionLargerThreshold(defaults.mediumPaddingThreshold() * psf.getMax());
        case(LOW):
            return psf.getRegionLargerThreshold(defaults.lowPaddingThreshold() * psf.getMax());
        case(MAX):
            return psf.getRegionLargerThreshold(1e-20f); // basically only zeroes are not included in padding
        case(NONE):
            return CuboidShape{0,0,0};
        default:
            assert(false);
            return CuboidShape{0,0,0};
    }
}

void PSFHandler::loadConfigsFromSetup(const SetupConfig& setupConfig) {
    if (configsLoaded) return;
    configsLoaded = true;

    psfConfigs = setupConfig.psfConfigs;

    if (!setupConfig.psfFilePaths.empty()) {
        std::vector<PSF> filePSFs = PSFCreator::readPSFsFromFilePath(setupConfig.psfFilePaths);
        for (auto& psf : filePSFs){
            psfs.push_back(std::make_shared<PSF>(psf));
        }
    }
}


float PSFHandler::generationCutoffThreshold(const DeconvolutionConfig& deconvConfig) {
    // The PSF is generated generationThresholdMultiplier below the trim threshold
    // so the later trim (padding) keeps a margin instead of cutting into signal.
    const auto& defaults = DefaultParameters::getDefaultParameters();
    if (deconvConfig.paddingRelativeMax > 0.0f)
        return deconvConfig.paddingRelativeMax * defaults.generationThresholdMultiplier();

    switch (deconvConfig.paddingStrategyType) {
        case(LOW):
            return defaults.lowPaddingThreshold() * defaults.generationThresholdMultiplier();
        case(MAX):
            return 1e-9f; // full support; the size cap bounds the cost
        case(DEFAULT):
            [[fallthrough]];
        case(NONE):
            [[fallthrough]];
        case(MEDIUM):
            return defaults.mediumPaddingThreshold() * defaults.generationThresholdMultiplier();
        default:
            return defaults.mediumPaddingThreshold() * defaults.generationThresholdMultiplier();
    }
}

void PSFHandler::generatePSFs(
    const SetupConfig& setupConfig,
    const DeconvolutionConfig& deconvConfig,
    const CuboidShape& maxSize)
{
    loadConfigsFromSetup(setupConfig);

    for (const auto& config : psfConfigs){
        if (config->sizeX == 0) config->sizeX = maxSize.width;
        if (config->sizeY == 0) config->sizeY = maxSize.height;
        if (config->sizeZ == 0) config->sizeZ = maxSize.depth;
        config->autoSize = true;
        if (config->cutoffThreshold <= 0.0f)
            config->cutoffThreshold = generationCutoffThreshold(deconvConfig);
        auto psf = std::make_shared<PSF>(PSFCreator::generatePSFFromPSFConfig(config, threadpool, progressFn));
        psfs.push_back(psf);
    }

    psfsGenerated = true;
}


Result<PaddingScheme> PSFHandler::getPadding(
    const DeconvolutionConfig& deconvConfig) const
{
    const bool cubePaddingSet = deconvConfig.cubePadding[0] >= 0 && deconvConfig.cubePadding[1] >= 0 && deconvConfig.cubePadding[2] >= 0;
    const bool relativeMaxSet = deconvConfig.paddingRelativeMax > 0.0f;

    // precedence: padding_relative_max > cube_padding > preset; log the decision once here
    if (relativeMaxSet) {
        if (cubePaddingSet) spdlog::get("deconvolution")->info("padding_relative_max {} and cube_padding both provided, using padding_relative_max", deconvConfig.paddingRelativeMax);
        if (deconvConfig.paddingStrategyType != PaddingStrategyType::DEFAULT) spdlog::get("deconvolution")->info("using padding_relative_max {} instead of preset {}", deconvConfig.paddingRelativeMax, paddingStrategyName(deconvConfig.paddingStrategyType));
        if (psfs.empty()) return Result<PaddingScheme>::fail("padding_relative_max requires at least one PSF");
        std::vector<CuboidShape> psfPaddings;
        for (const auto& psf : psfs) psfPaddings.push_back(getManualRelativeMaxPadding(*psf, deconvConfig.paddingRelativeMax));
        return makePaddingScheme(getLargestShape(psfPaddings));
    }

    if (cubePaddingSet) {
        if (deconvConfig.paddingStrategyType != PaddingStrategyType::DEFAULT) spdlog::get("deconvolution")->info("using cube_padding [{} {} {}] instead of preset {}", deconvConfig.cubePadding[0], deconvConfig.cubePadding[1], deconvConfig.cubePadding[2], paddingStrategyName(deconvConfig.paddingStrategyType));
        return makePaddingScheme(getManualCubePadding(deconvConfig.cubePadding));
    }

    if (deconvConfig.paddingStrategyType == PaddingStrategyType::DEFAULT){
        spdlog::get("deconvolution")->info("No padding_strategy provided, using preset MEDIUM");
    }
    if (deconvConfig.cubePadding[0] >= 0 || deconvConfig.cubePadding[1] >= 0 || deconvConfig.cubePadding[2] >= 0){
        spdlog::get("deconvolution")->warn("cube_padding partially provided [{} {} {}], ignoring, falling back to preset {}", deconvConfig.cubePadding[0], deconvConfig.cubePadding[1], deconvConfig.cubePadding[2], paddingStrategyName(deconvConfig.paddingStrategyType));
    }
    if (psfs.empty()){
        return Result<PaddingScheme>::fail("No PSFs available, cannot determine padding");
    }
    std::vector<CuboidShape> psfPaddings;
    for (const auto& psf : psfs){
        psfPaddings.push_back(getPresetPadding(*psf, deconvConfig.paddingStrategyType));
    }
    return makePaddingScheme(getLargestShape(psfPaddings));
}


Result<PaddingScheme> PSFHandler::makePaddingScheme(CuboidShape totalPadding) const {
    Padding padding{totalPadding / 2, totalPadding - totalPadding / 2};

    PaddingScheme scheme;
    scheme.insidePadding = padding;
    //TODO which imagepadding, which cubepadding
    scheme.imagePadding.before = scheme.insidePadding.before; /// 2;
    scheme.imagePadding.after = scheme.insidePadding.after; /// 2;

    spdlog::get("deconvolution")->debug("PSFHandler reported the following necessary padding: inside {} before, {} after; image {} before, {} after",
        scheme.insidePadding.before.print(), scheme.insidePadding.after.print(),
        scheme.imagePadding.before.print(), scheme.imagePadding.after.print());

    assert(scheme.insidePadding.before >= scheme.imagePadding.before); // currently needed by imagesplit
    return Result<PaddingScheme>::ok(std::move(scheme));
}


CuboidShape PSFHandler::getMaxShape() const
{
    std::vector<CuboidShape> psfShapes;

    for (const auto& psf : psfs){
        psfShapes.push_back(psf->getShape());
    }

    CuboidShape largestPSF = getLargestShape(psfShapes);

    return largestPSF;
}

std::vector<std::shared_ptr<PSF>>& PSFHandler::getPSFs(const CuboidShape& shape){
    for (std::shared_ptr<PSF> psf: psfs){
        ImagePadding::fitToShape(*psf, shape, PaddingFillType::ZERO);
    }
    return psfs;
}


std::unique_ptr<PSFPreprocessor> PSFHandler::createPSFPreprocessor() const {

    std::function<std::unique_ptr<ComplexData>(const CuboidShape, std::shared_ptr<PSF>, IBackend&)> psfPreprocessFunction = [&](
        const CuboidShape targetShape,
        std::shared_ptr<PSF> inputPSF,
        IBackend& backend
    ) -> std::unique_ptr<ComplexData>
        {
            auto logger = spdlog::get("deconvolution");
            logger->debug("Preprocessing PSF...");
            if (targetShape != inputPSF->getShape()) throw std::runtime_error("PSF Preprocessor: PSF in shape " + inputPSF->getShape().print() + " but should be in shape " + targetShape.print());

            RealData h = Preprocessor::convertImageToRealData(*inputPSF);
            RealData h_device = backend.getMemoryManager().copyDataToDevice(h);

            std::unique_ptr<ComplexView> h_result_device = std::make_unique<ComplexView>(std::move(backend.getMemoryManager().reinterpret(h_device)));
            backend.getComputeManager().octantFourierShift(h_device); // align psf peak at 0,0,0

            backend.getComputeManager().forwardFFT(h_device, *h_result_device);

            h_result_device->setBackend(h_device.getBackend());
            h_device.setBackend(nullptr);

            backend.sync();
            return std::move(h_result_device);
        };

    std::unique_ptr<PSFPreprocessor> preprocessor = std::make_unique<PSFPreprocessor>();
    preprocessor->setPreprocessingFunction(psfPreprocessFunction);
    return std::move(preprocessor);
}
