#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/PSFCreator.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include <spdlog/spdlog.h>


CuboidShape PSFHandler::getPSFPadding(const PSF& psf, PaddingStrategyType paddingType, float paddingRelativeMax) const {
    CuboidShape padding;
    switch(paddingType){
    case(PARENT):
        padding = PaddingStrategy::parentPadding(psf, paddingRelativeMax);
        break;
    case(FULL_PSF):
        padding = PaddingStrategy::fullPSFPadding(psf);
        break;
    default:
        padding = CuboidShape{0, 0, 0};
        break;
    }
    return padding;

}



CuboidShape PSFHandler::getPaddingFromConfig(std::shared_ptr<PSFConfig> config, PaddingStrategyType paddingStrategy) const {
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<BasePSFGenerator> psfGenerator = factory.createGenerator(config);
    return psfGenerator->getPadding(paddingStrategy);
}


void PSFHandler::loadConfigsFromSetup(const SetupConfig& setupConfig) {
    if (configsLoaded) return;
    configsLoaded = true;

    if (hasInlineConfigs()) {
        psfConfigs = inlinePsfConfigs;
    }

    if (!setupConfig.psfFilePaths.empty()) {
        filePSFs = PSFCreator::readPSFsFromFilePath(setupConfig.psfFilePaths);
    }
}


Result<Padding> PSFHandler::getPadding(
    const SetupConfig& setupConfig,
    const DeconvolutionConfig& deconvConfig)
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
            loadConfigsFromSetup(setupConfig);

            std::vector<CuboidShape> psfPaddings;

            for (const auto& config : psfConfigs){
                CuboidShape paddingSize = getPaddingFromConfig(config, deconvConfig.paddingStrategyType);
                psfPaddings.push_back(paddingSize);
            }

            for (auto& psf : filePSFs){
                CuboidShape paddingSize = getPSFPadding(psf, deconvConfig.paddingStrategyType, deconvConfig.paddingRelativeMax);
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




Result<CuboidShape> PSFHandler::getMaxShape(
    const SetupConfig& setupConfig,
    const DeconvolutionConfig& deconvConfig)
{
    loadConfigsFromSetup(setupConfig);

    std::vector<CuboidShape> psfShapes;

    for (const auto& config : psfConfigs){
        psfShapes.push_back(config->getShape());
    }

    for (auto& psf : filePSFs){
        psfShapes.push_back(psf.getShape());
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

    std::vector<std::shared_ptr<PSF>> psfs;

    for (auto& config : psfConfigs){
        config->sizeX = psfShape.width;
        config->sizeY = psfShape.height;
        config->sizeZ = psfShape.depth;
        std::shared_ptr<PSF> psf = std::make_shared<PSF>(PSFCreator::generatePSFFromPSFConfig(config, threadpool, progressFn));
        auto logger = spdlog::get("config");
        logger->debug("Using the following PSF from config");
        config->printValues();
        psfs.emplace_back(psf);

    }

    for (auto& psf : filePSFs){
        psfs.emplace_back(std::make_shared<PSF>(std::move(psf)));
    }

    if (psfs.size() <= 0){
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
