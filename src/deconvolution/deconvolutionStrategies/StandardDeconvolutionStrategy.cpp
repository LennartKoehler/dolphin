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

#include "dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "dolphin/ProgressTracking.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include <cmath>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin_image/HelperClasses.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/PSFCreator.h"
#include "dolphin/psf/PSFGeneratorFactory.h"



Result<DeconvolutionPlan> StandardDeconvolutionStrategy::createPlan(
    std::shared_ptr<ImageReader> reader,
    std::shared_ptr<ImageWriter> writer,
    PSFHandler& psfHandler,
    const DeconvolutionConfig& deconvConfig,
    const SetupConfig& setupConfig
) {

    Memory memory = resolveMemory(setupConfig);

    // set up reader and writer
    ReaderConfig readerConfig;
    readerConfig.numReaderThreads = setupConfig.numReaderThreads > 0
        ? static_cast<size_t>(setupConfig.numReaderThreads)
        : setupConfig.nIOThreads;
    readerConfig.readerMemory_byte = memory.hostMem_byte;
    int readerChannel = 0; //unused
    reader->configure(readerChannel, readerConfig);
    std::shared_ptr<ReaderHandler> readerHandler = std::make_shared<ReaderHandler>(reader, deconvConfig.paddingFillType);

    WriterCompressionConfig writerConfig;
    writerConfig.compressionScheme = WriterCompressionConfig::parseCompression(setupConfig.outputCompression);
    writerConfig.compressionLevel = setupConfig.outputCompressionLevel;
    writer->configure(writerConfig);
    // -----------------------



    ImageMetaData metadata = reader->getMetaData();
    CuboidShape imageSize = CuboidShape{metadata.imageWidth, metadata.imageLength, metadata.slices};

    std::shared_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(deconvConfig);

    spdlog::get("deconvolution")->debug("Using the following deconvolution config");
    deconvConfig.printValues();

    spdlog::get("deconvolution")->debug("Using the following setup config");
    setupConfig.printValues();


    size_t totalThreads = setupConfig.nThreads;
    size_t ioThreads = setupConfig.nIOThreads;
    size_t workerThreads = setupConfig.nWorkerThreads;
    int nDevices = setupConfig.nDevices;
    // these are necessary as the cpu can have actual seperate threads for each task or use multiple threads for one task
    BackendConfig workerConfig; // these are configured by the backend so that when its given a task it know what to do
    BackendConfig ioConfig;


    resolveThreadsAndDevices(
        BackendFactory::getBackendManagerStatic(setupConfig.backend),
		nDevices,
		workerThreads,
		ioThreads,
		totalThreads,
        workerConfig,
        ioConfig);


    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = getCubes(
        ioThreads,
        workerThreads,
        memory.deviceMem_byte,
        nDevices,
        algorithm,
        psfHandler,
        deconvConfig,
        setupConfig,
        imageSize);


    std::vector<std::shared_ptr<TaskContext>> contexts = createContexts(
        BackendFactory::getBackendManagerStatic(setupConfig.backend),
        psfHandler,
		setupConfig.nDevices,
		workerThreads,
		ioThreads,
        ioConfig,
        workerConfig);

    BoxCoordWithPadding workShape = cubeCoordinatesWithPadding[0]; // the final and actual shape with padding that will be used
    Padding padding = workShape.padding;

    std::vector<std::shared_ptr<PSF>> psfs = psfHandler.createPSFs(workShape.getPaddedShape());

    for (const auto& psf : psfs){
        // this is for the psfs provided as files, the other ones are created using the cube size (see 2 lines above)
        // the read psfs that are SMALLER are later padded to that shape
        if (workShape.getPaddedShape() < psf->getShape()){
             // spdlog::get("deconvolution")->critical("PSF (ID: {} size: ({})) is larger than the maximum subimage shape ({}) for the given memory limits", psf->getShape().print(), workShape.box.dimensions.print());
            throw std::runtime_error("PSF (ID: {} size: ({})) is larger than the maximum subimage shape ({}) for the given memory limits");
        }
    }

    if (workShape.padding.getTotalPadding() / workShape.box.dimensions> 3)
        spdlog::get("deconvolution")->warn("Low memory, padding takes up most of the compute block. Padding is: ({}); the subimage is: ({})",
                                           padding.getTotalPadding().print(), workShape.box.dimensions.print());

    size_t memoryPerTask = estimateMemoryUsage(workShape.getPaddedBox().dimensions, algorithm.get(), setupConfig);

    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());

    std::shared_ptr<CubeTaskSharedDescriptor> sharedTaskDescriptor = std::make_unique<CubeTaskSharedDescriptor>(
            algorithm,
            memoryPerTask,
            psfs,
            readerHandler,
            writer);

    for (size_t i = 0; i < cubeCoordinatesWithPadding.size(); ++i) {

        std::shared_ptr<TaskContext> context = contexts[i % contexts.size()]; // cycle through contexts and assign the context to that task
        // TODO if the contexts are uneven in their compute capability then this would have to be more dynamic,
        // one idea would be not to assign the context here at all, and just attach the contexts to the deconvplan, not the individiaul tasks
        // then the deconvexecutor could dynamically attach contexts to tasks when that context is running out of tasks,
        // so like having a queue and only filling each context queue when that queue is getting empty, then slow queues wouldnt get that much work

        tasks.push_back(std::make_unique<CubeTaskDescriptor>(
            static_cast<int>(i),
            cubeCoordinatesWithPadding[i],
            sharedTaskDescriptor,
            context));
    }

    size_t totalTasks = tasks.size();
    assert (totalTasks > 0 && "No tasks in deconvolution plan");

    spdlog::get("deconvolution")
        ->info("Successfully created deconvolution plan with {} total cubes. Each cube has size (width x height x depth) ({}) which includes padding (padding before, padding after) ({}, {})",
        totalTasks, (workShape.getPaddedBox().dimensions).print(), workShape.padding.before.print(), workShape.padding.after.print());

    DeconvolutionPlan plan {
        std::move(tasks),
        totalTasks
    };
    return std::move(Result<DeconvolutionPlan>::ok(std::move(plan)));
}


std::vector<BoxCoordWithPadding> StandardDeconvolutionStrategy::getCubes(
    const size_t& ioThreads,
    const size_t& workerThreads,
    const size_t& maxMemDevice_byte,
    const int& nDevices,
    std::shared_ptr<DeconvolutionAlgorithm> algorithm,
    PSFHandler& psfHandler,
    const DeconvolutionConfig& deconvConfig,
    const SetupConfig& setupConfig,
    const CuboidShape& imageSize
) const {
    size_t maxMemoryPerCube = getMaxMemoryPerCube(
        ioThreads,
        workerThreads,
        maxMemDevice_byte,
        BackendFactory::getBackendManagerStatic(setupConfig.backend).createBackendForCurrentThread(BackendConfig()).getMemoryManager(),
        algorithm
    );
    size_t maxMemCubeVolume = maxMemoryPerCube / sizeof(real_t);

    Result<Padding> paddingResult = psfHandler.getPadding(setupConfig, deconvConfig, imageSize);
    // this cubepadding might still change due to good shapes for DFT. But this is the minimum!
    if (!paddingResult.success) {
        throw std::runtime_error("Error while getting Padding");
    }
    Padding padding = std::move(paddingResult.value);

    if(padding.before + padding.after < deconvConfig.featheringRadius)
        spdlog::get("deconvolution")->warn("Feathering radius ({}) is smaller than padding (which is probably the size of the psf) ({}), which can cause artifacts",
            deconvConfig.featheringRadius, (padding.before + padding.after).print());

    // Padding imagePadding = getImagePadding(imageSize, idealCubeSize.value, padding);

    int maxSubCubes = 20;
    CuboidShape minShape = imageSize / maxSubCubes + padding.getTotalPadding() + CuboidShape{1,1,1}; // TESTVALUE "max of 10 subcubes"

    Result<std::vector<BoxCoordWithPadding>> cubeCoordinatesWithPaddingResult = splitImageHomogeneous(padding, imageSize, maxMemCubeVolume, workerThreads * nDevices, deconvConfig.paddingStrategyType, minShape);
    if (!cubeCoordinatesWithPaddingResult.success) {
        throw std::runtime_error("Error while splitting image");
    }
    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = cubeCoordinatesWithPaddingResult.value;
    return cubeCoordinatesWithPadding;
}


void StandardDeconvolutionStrategy::resolveThreadsAndDevices(
    IBackendManager& manager,
    int& configNDevices,
    size_t& nWorkerThreads,
    size_t& nIOThreads,
    size_t& totalThreads,
    BackendConfig& ioconfig,
    BackendConfig& workerconfig
) const{
        int numberDevices = manager.getNumberDevices();
        numberDevices = std::min(numberDevices, configNDevices);
        numberDevices = numberDevices < 1 ? 1 : numberDevices;

        // TODO the backendmanager knows best the ratio between workers and io, however this is not really a responsibility of the backend
        // so this is like a question to backend: How would you use these number of io and worker threads
        // and then this implicitly can also impact the number of threadpools used in the execution as the total number of threads shouldnt be larger than what is in the context
        // then the backendconfig which got incluenced by the manager is passed to the taskcontext, where it is later used for the same manager
        // to init new backends
        // so nIOThreads and nWorkerThreads are the number of threads in the respective threadpool
        // while the backendconfigs might e.g. the number of ompbackends
        manager.setThreadDistribution(totalThreads, nIOThreads, nWorkerThreads, ioconfig, workerconfig);
}

std::vector<std::shared_ptr<TaskContext>> StandardDeconvolutionStrategy::createContexts(
    IBackendManager& manager,
    PSFHandler& psfHandler,
    int numberDevices,
    const size_t& nWorkerThreads,
    const size_t& nIOThreads,
    BackendConfig ioconfig,
    BackendConfig workerconfig
) const{


    std::vector<std::shared_ptr<TaskContext>> contexts;

    for (int i = 0; i < numberDevices; i++){
        BackendConfig ctxIoConfig = ioconfig;
        BackendConfig ctxWorkerConfig = workerconfig;
        ctxIoConfig.deviceId = i;
        ctxWorkerConfig.deviceId = i;

        std::shared_ptr<TaskContext> context = std::make_shared<TaskContext>(manager, ctxIoConfig, ctxWorkerConfig, nWorkerThreads, nIOThreads);

        std::unique_ptr<PSFPreprocessor> preprocessor = psfHandler.createPSFPreprocessor();
        context->setPreprocessor(std::move(preprocessor));
        contexts.emplace_back(context);
    }
    return contexts;
}


std::shared_ptr<DeconvolutionAlgorithm> StandardDeconvolutionStrategy::getAlgorithm(const DeconvolutionConfig& config) {
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = fact.createShared(config);
    return algorithm;
}



size_t StandardDeconvolutionStrategy::getMaxMemoryPerCube(
    size_t ioThreads,
    size_t workerThreads,
    size_t maxMemory,
    const IBackendMemoryManager& backend,
    std::shared_ptr<DeconvolutionAlgorithm> algorithm
) const {

    FFTWorkspaceCopiesEstimator estimator = [&backend](const CuboidShape& shape) {
        return backend.estimateFFTWorkspaceCopies(shape);
    };

    return computeMaxMemoryPerCube(
        maxMemory,
        ioThreads,
        workerThreads,
        algorithm->getMemoryMultiplier(),
        estimator);
}

size_t StandardDeconvolutionStrategy::computeMaxMemoryPerCube(
    size_t availableMemory,
    size_t ioThreads,
    size_t workerThreads,
    size_t algorithmMemoryMultiplier,
    const FFTWorkspaceCopiesEstimator& estimateFFTWorkspaceCopies
){

    size_t workerAllocations = workerThreads * algorithmMemoryMultiplier;

    int ioCopies = 3; //image, psf, result, but psf only allocated once in total
    size_t ioAllocations = ioThreads * ioCopies;

    size_t threadallocations = ioAllocations + workerAllocations;
    assert(threadallocations != 0 && "Error, no threadallocations");
    size_t memoryPerCube = availableMemory / threadallocations;

    int side = static_cast<int>(std::cbrt(static_cast<double>(memoryPerCube) / sizeof(real_t)));
    if (side < 1) side = 1;
    side = nextSmooth(side);
    CuboidShape estimatedShape(side, side, side);

    float fftwWorkspaceCopies = estimateFFTWorkspaceCopies(estimatedShape);

    memoryPerCube = availableMemory / (fftwWorkspaceCopies + threadallocations);

    return memoryPerCube;
}



size_t StandardDeconvolutionStrategy::estimateMemoryUsage(
    const CuboidShape& cubeSize,
    const DeconvolutionAlgorithm* algorithm,
    const SetupConfig& config
){
    int ioCopies = 3; //image, psf, result
    size_t ioAllocations = config.nIOThreads * cubeSize.getVolume() * ioCopies * sizeof(real_t);
    size_t workerAllocations = config.nWorkerThreads * cubeSize.getVolume() * algorithm->getMemoryMultiplier() * sizeof(complex_t);
    return ioAllocations + workerAllocations;
}




// void StandardDeconvolutionStrategy::configureReaderWriter(
//     std::shared_ptr<ImageReader> reader,
//     std::shared_ptr<ImageWriter> writer,
//     size_t numReaderThreads,
//     size_t readerMemory,
//     size_t numWriterThreads,
//     size_t writerMemory,
// ){
// }



Memory StandardDeconvolutionStrategy::resolveMemory(const SetupConfig& config) const{

    Memory memory;
    size_t maxMemDevice_byte = static_cast<size_t>(config.maxMemDevice_gb * 1024 * 1024 * 1024);
    size_t maxMemHost_byte = static_cast<size_t>(config.maxMemHost_gb * 1024 * 1024 * 1024);
    //memDevice is also synonym for memory on backend, in the case where device == host the deconvstrategy will use maxMemDevice to set up the deconvolutionstrategy
    if (config.backend == HOST_BACKEND){
        size_t totalMemoryOnHost = std::max(maxMemDevice_byte, maxMemHost_byte);
        totalMemoryOnHost = std::max(totalMemoryOnHost, maxMemDevice_byte + maxMemHost_byte);
        totalMemoryOnHost = std::min(totalMemoryOnHost, BackendFactory::getHostBackendMemoryManagerStatic().getAvailableMemory());
        totalMemoryOnHost = totalMemoryOnHost <= 0 ? BackendFactory::getHostBackendMemoryManagerStatic().getAvailableMemory() : totalMemoryOnHost;
        memory.hostMem_byte = 1/3.5 * totalMemoryOnHost; // need a bit for the io copy operations, dont use everything
        memory.deviceMem_byte = 2/3.5 * totalMemoryOnHost;


    }
    else{
        // on device and host use everything under the limit for device and host
        // assumption is that if a cube fits on the device (e.g. gpu) then it will also fit on RAM
        const IBackendMemoryManager& deviceManager = BackendFactory::getBackendManagerStatic(config.backend).createBackendForCurrentThread(BackendConfig()).getMemoryManager();
        memory.deviceMem_byte = maxMemDevice_byte <= 0 ? deviceManager.getAvailableMemory() : maxMemDevice_byte;
        memory.deviceMem_byte = std::min(memory.deviceMem_byte, deviceManager.getAvailableMemory());
        memory.hostMem_byte = maxMemHost_byte <= 0 ? BackendFactory::getHostBackendMemoryManagerStatic().getAvailableMemory() : maxMemHost_byte;
        memory.hostMem_byte = std::min(memory.hostMem_byte, BackendFactory::getHostBackendMemoryManagerStatic().getAvailableMemory());
    }

    return memory;
}
