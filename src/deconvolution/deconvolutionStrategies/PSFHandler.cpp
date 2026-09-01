#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/PSFCreator.h"
#include <spdlog/spdlog.h>


CuboidShape PSFHandler::getPSFPadding(const PSF& psf, PaddingStrategyType paddingType, float paddingRelativeMax) const {
    CuboidShape padding;
    switch(paddingType){
    case(PARENT): {
        PSFExtent extent = psf.computeEnergyExtent(0.98, 0.98);
        padding = CuboidShape{extent.lateralExtent, extent.lateralExtent, extent.zHalfExtent};
        break;
    }
    case(FULL_PSF):
        padding = psf.getShape();
        break;
    default:
        padding = CuboidShape{0, 0, 0};
        break;
    }
    return padding;
}


void PSFHandler::loadConfigsFromSetup(const SetupConfig& setupConfig) {
    if (configsLoaded) return;
    configsLoaded = true;

    if (hasInlineConfigs()) {
        psfConfigs = inlinePsfConfigs;
    }

    if (!setupConfig.psfFilePaths.empty()) {
        std::vector<PSF> filePSFs = PSFCreator::readPSFsFromFilePath(setupConfig.psfFilePaths);
        for (auto& psf : filePSFs){
            psfs.push_back(std::make_shared<PSF>(psf));
        }
    }
}


void PSFHandler::generatePSFs(
    const SetupConfig& setupConfig,
    const CuboidShape& maxSize)
{
    loadConfigsFromSetup(setupConfig);

    for (const auto& config : psfConfigs){
        if (config->sizeX == 0) config->sizeX = maxSize.width;
        if (config->sizeY == 0) config->sizeY = maxSize.height;
        if (config->sizeZ == 0) config->sizeZ = maxSize.depth;
        auto psf = std::make_shared<PSF>(PSFCreator::generatePSFFromPSFConfig(config, threadpool, progressFn));
        psfs.push_back(psf);
    }

    psfsGenerated = true;
}


Result<Padding> PSFHandler::getPadding(
    const DeconvolutionConfig& deconvConfig) const
{
    Padding padding;

    switch(deconvConfig.paddingStrategyType){
        case NONE:{
            padding = Padding{CuboidShape{0,0,0}, CuboidShape{0,0,0}};
            break;
        }
        case MANUAL:{
            CuboidShape manualPadding{
                static_cast<size_t>(std::max(0, deconvConfig.cubePadding[0])),
                static_cast<size_t>(std::max(0, deconvConfig.cubePadding[1])),
                static_cast<size_t>(std::max(0, deconvConfig.cubePadding[2]))
            };
            padding = Padding{manualPadding / 2, manualPadding - manualPadding / 2};
            break;
        }

        default:{
            std::vector<CuboidShape> psfPaddings;
            for (const auto& psf : psfs){
                CuboidShape paddingSize = getPSFPadding(*psf, deconvConfig.paddingStrategyType, deconvConfig.paddingRelativeMax);
                psfPaddings.push_back(paddingSize);
            }

            CuboidShape result = getLargestShape(psfPaddings);
            padding = Padding{result / 2, result - result / 2};
        }
    }

    if (padding.before < CuboidShape{0,0,0} ||
        padding.after  < CuboidShape{0,0,0})
    {
        return Result<Padding>::fail(
            "Padding for cubes is smaller than zero");
    }


    return Result<Padding>::ok(std::move(padding));
}


Result<CuboidShape> PSFHandler::getMaxShape() const
{
    std::vector<CuboidShape> psfShapes;

    for (const auto& psf : psfs){
        psfShapes.push_back(psf->getShape());
    }

    CuboidShape largestPSF = getLargestShape(psfShapes);

    if (largestPSF < CuboidShape{0,0,0})
    {
        return Result<Padding>::fail(
            "Padding for cubes is smaller than zero");
    }


    return Result<CuboidShape>::ok(std::move(largestPSF));
}

std::vector<std::shared_ptr<PSF>> PSFHandler::createPSFs(
    const CuboidShape& psfShape)
{
    if (!psfsGenerated) {
        for (const auto& config : psfConfigs){
            if (config->sizeX == 0) config->sizeX = psfShape.width;
            if (config->sizeY == 0) config->sizeY = psfShape.height;
            if (config->sizeZ == 0) config->sizeZ = psfShape.depth;
            auto psf = std::make_shared<PSF>(PSFCreator::generatePSFFromPSFConfig(config, threadpool, progressFn));
            psfs.push_back(psf);
        }
        psfsGenerated = true;
    }

    for (auto& psf : psfs){
        CuboidShape currentShape = psf->getShape();
        if (currentShape < psfShape) {
            ImagePadding::padToShape(*psf, psfShape, PaddingFillType::ZERO);
        } else if (psfShape < currentShape) {
            spdlog::get("deconvolution")->critical("PSF (size: {}) is larger than the cube shape ({})", currentShape.print(), psfShape.print());
            throw std::runtime_error("PSF too large for cube constraints");
        }
    }

    if (psfs.empty()){
        throw std::runtime_error("No PSFs supplied as either a PSF Config or as a file");
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

            ImagePadding::padToShape(*inputPSF, targetShape, PaddingFillType::ZERO);
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
