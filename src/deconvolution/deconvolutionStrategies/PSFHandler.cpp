#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/PSFCreator.h"
#include "dolphin/psf/PSFGeneratorFactory.h"

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
        padding = CuboidShape{-1, -1, -1};
        break;
    }
    return padding;

}



CuboidShape PSFHandler::getPaddingFromConfig(std::shared_ptr<PSFConfig> config, PaddingStrategyType paddingStrategy) const {
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<BasePSFGenerator> psfGenerator = factory.createGenerator(config);
    // the specific generator should know best how much should be padded without actually computing the psf
    // e.g. the gaussian generator can say for a given strategy you should pad up to sigma * 2
    return psfGenerator->getPadding(paddingStrategy);
}


Result<Padding> PSFHandler::getPadding(
    const SetupConfig& setupConfig,
    const DeconvolutionConfig& deconvConfig,
    const CuboidShape& imageSize)
{
    Padding padding;

    switch(deconvConfig.paddingStrategyType){
        case NONE:{
            padding = Padding{CuboidShape{0,0,0}, CuboidShape{0,0,0}};
            break;
        }
        case MANUAL:{
            CuboidShape manualPadding = CuboidShape(deconvConfig.cubePadding);
            padding = Padding{manualPadding / 2, manualPadding - manualPadding / 2};
            break;
        }

        default:{
            std::vector<CuboidShape> psfPaddings;

            // if a config is given and the psf is generated here then use the psfgenerator to determine how much should
            // be padded before actually creating the psf, then when creating the psf create it with the size of the image plus padding
            if (!setupConfig.multiplePsfConfigPaths.empty()){
                psfConfigs = PSFCreator::generatePSFConfigsFromConfigPath(setupConfig.multiplePsfConfigPaths);
                for (const auto& config : psfConfigs){
                    CuboidShape paddingSize = getPaddingFromConfig(config, deconvConfig.paddingStrategyType);
                    psfPaddings.push_back(paddingSize);
                }
            }

            // if psfs are given as files, then the padding is determined by a a function e.g.
            // just a fixed size or up until the values are below a threshold
            if (!setupConfig.psfFilePaths.empty()){
                filePSFs = PSFCreator::readPSFsFromFilePath(setupConfig.psfFilePaths);
                for (auto& psf : filePSFs){
                    CuboidShape paddingSize = getPSFPadding(psf, deconvConfig.paddingStrategyType, deconvConfig.paddingRelativeMax);
                    psfPaddings.push_back(paddingSize);
                }
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

    // assert(padding.getTotalPadding() + imageSize >= largestPSF);

    return Result<Padding>::ok(std::move(padding));
}


std::vector<std::shared_ptr<PSF>> PSFHandler::createPSFs(
    const CuboidShape& psfShape)
{

    std::vector<std::shared_ptr<PSF>> psfs;

    for (auto& config : psfConfigs){
        config->sizeX = psfShape.width;
        config->sizeY = psfShape.height;
        config->sizeZ = psfShape.depth;
        psfs.emplace_back(std::make_shared<PSF>(PSFCreator::generatePSFFromPSFConfig(config, threadpool, progressFn)));
    }

    //dont need to reread, already read when getting padding
    for (auto& psf : filePSFs){
        psfs.emplace_back(std::make_shared<PSF>(std::move(psf)));
    }

    // TODO
    // if (!setupConfig.psfDirPath.empty()){
    //     std::vector<std::shared_ptr<PSFConfig>> psfconfigs = PSFCreator::generatePSFsFromDir(setupConfig.psfDirPath);
    //     for (auto psfconfig : psfconfigs){
    //         psfs.push_back(PSFCreator::generatePSFFromPSFConfig(psfconfig, thread_pool_.get()));
    //     }
    // }
    return psfs;
}


std::unique_ptr<PSFPreprocessor> PSFHandler::createPSFPreprocessor() const {

    std::function<std::unique_ptr<ComplexData>(const CuboidShape, std::shared_ptr<PSF>, IBackend&)> psfPreprocessFunction = [&](
        const CuboidShape targetShape,
        std::shared_ptr<PSF> inputPSF,
        IBackend& backend
    ) -> std::unique_ptr<ComplexData>
        {
            Preprocessor::padToShape(*inputPSF, targetShape, PaddingFillType::ZERO);
            RealData h = Preprocessor::convertImageToRealData(*inputPSF);
            RealData h_device = backend.getMemoryManager().copyDataToDevice(h);
            std::unique_ptr<ComplexView> h_result_device = std::make_unique<ComplexView>(std::move(backend.getMemoryManager().reinterpret(h_device)));
            backend.getDeconvManager().octantFourierShift(h_device); // align psf peak at 0,0,0

            backend.getDeconvManager().forwardFFT(h_device, *h_result_device);

            //transfer ownership of data
            h_result_device->setBackend(h_device.getBackend());
            h_device.setBackend(nullptr); // so basically now h_result_data owns the data and h_device no longer does because it doesnt have a backend to delete it

            // backend.getDeconvManager().backwardFFT(*h_result_device, h_device);

            // move back to host for cuda

            // RealData result = backend.getMemoryManager().moveDataFromDevice(h_device, BackendFactory::getInstance().getDefaultBackendMemoryManager());
            // Image3D test = Preprocessor::convertComplexDataToImage(*h_result_device);
            // TiffWriter::writeToFile("/home/lennart-k-hler/data/dolphin_results/psf_fft.tif", test);


            backend.sync();
            return std::move(h_result_device);
        };

    std::unique_ptr<PSFPreprocessor> preprocessor = std::make_unique<PSFPreprocessor>();
    preprocessor->setPreprocessingFunction(psfPreprocessFunction);
    return std::move(preprocessor);
}
