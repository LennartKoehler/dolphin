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

    ImageMetaData metadata = reader->getMetaData();
    CuboidShape imageSize = CuboidShape{metadata.imageWidth, metadata.imageLength, metadata.slices};
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(deconvConfig);

    spdlog::get("deconvolution")->debug("Using the following deconvolution config");
    deconvConfig.printValues();

    spdlog::get("deconvolution")->debug("Using the following setup config");
    setupConfig.printValues();

    IBackendManager& manager = getBackendManager(setupConfig);

    size_t totalThreads = setupConfig.nThreads;
    size_t ioThreads = setupConfig.nIOThreads;
    size_t workerThreads = setupConfig.nWorkerThreads;

    std::vector<std::shared_ptr<TaskContext>> contexts = createContexts(
        manager,
        psfHandler,
		setupConfig.nDevices,
		workerThreads,
		ioThreads,
		totalThreads);

    size_t maxMemoryPerCube = getMaxMemoryPerCube(
        ioThreads,
        workerThreads,
        manager.createBackendForCurrentThread(BackendConfig()).getMemoryManager(),
        algorithm
    );
    size_t maxMemCubeVolume = maxMemoryPerCube / sizeof(real_t);

    Result<Padding> paddingResult = psfHandler.getPadding(setupConfig, deconvConfig, imageSize);
    // this cubepadding might still change due to good shapes for DFT. But this is the minimum!
    if (!paddingResult.success) {
        return Result<DeconvolutionPlan>(paddingResult);
    }
    Padding padding = std::move(paddingResult.value);

    if(padding.before + padding.after < deconvConfig.featheringRadius)
        spdlog::get("deconvolution")->warn("Feathering radius ({}) is smaller than padding (which is probably the size of the psf) ({}), which can cause artifacts",
            deconvConfig.featheringRadius, (padding.before + padding.after).print());

    // Padding imagePadding = getImagePadding(imageSize, idealCubeSize.value, padding);

    int maxSubCubes = 10;
    CuboidShape minShape = imageSize / maxSubCubes + padding.getTotalPadding(); // TESTVALUE "max of 10 subcubes"

    Result<std::vector<BoxCoordWithPadding>> cubeCoordinatesWithPaddingResult = splitImageHomogeneous(padding, imageSize, maxMemCubeVolume, workerThreads, deconvConfig.paddingStrategyType, minShape);
    if (!cubeCoordinatesWithPaddingResult.success) {
        return Result<DeconvolutionPlan>(cubeCoordinatesWithPaddingResult);
    }
    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = cubeCoordinatesWithPaddingResult.value;

    BoxCoordWithPadding workShape = cubeCoordinatesWithPadding[0]; // the final and actual shape with padding that will be used

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

    size_t memoryPerTask = estimateMemoryUsage(workShape.getBox().dimensions, algorithm.get(), setupConfig);

    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());

    std::shared_ptr<CubeTaskSharedDescriptor> sharedTaskDescriptor = std::make_unique<CubeTaskSharedDescriptor>(
            algorithm,
            memoryPerTask,
            psfs,
            reader,
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
        totalTasks, (workShape.getBox().dimensions).print(), workShape.padding.before.print(), workShape.padding.after.print());

    DeconvolutionPlan plan {
        std::move(tasks),
        totalTasks
    };
    return std::move(Result<DeconvolutionPlan>::ok(std::move(plan)));
}




std::vector<std::shared_ptr<TaskContext>> StandardDeconvolutionStrategy::createContexts(
    IBackendManager& manager,
    PSFHandler& psfHandler,
    int configNDevices,
    size_t& nWorkerThreads,
    size_t& nIOThreads,
    size_t& totalThreads) const
{
        int numberDevices = manager.getNumberDevices();
        numberDevices = std::min(numberDevices, configNDevices);
        numberDevices = numberDevices < 1 ? 1 : numberDevices;

        BackendConfig ioconfig;
        BackendConfig workerconfig;
        // TODO the backendmanager knows best the ratio between workers and io, however this is not really a responsibility of the backend
        // so this is like a question to backend: How would you use these number of io and worker threads
        // and then this implicitly can also impact the number of threadpools used in the execution as the total number of threads shouldnt be larger than what is in the context
        // then the backendconfig which got incluenced by the manager is passed to the taskcontext, where it is later used for the same manager
        // to init new backends
        // so nIOThreads and nWorkerThreads are the number of threads in the respective threadpool
        // while the backendconfigs might e.g. the number of ompbackends
        manager.setThreadDistribution(totalThreads, nIOThreads, nWorkerThreads, ioconfig, workerconfig);


        std::vector<std::shared_ptr<TaskContext>> contexts;

        for (int i = 0; i < numberDevices; i++){
            std::shared_ptr<TaskContext> context = std::make_shared<TaskContext>(manager, ioconfig, workerconfig, nWorkerThreads, nIOThreads);

            std::unique_ptr<PSFPreprocessor> preprocessor = psfHandler.createPSFPreprocessor(); // new psfpreprocessor for each context because the psfs live on devicecudabac
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

IBackendManager& StandardDeconvolutionStrategy::getBackendManager(const SetupConfig& config){
    BackendFactory& bf = BackendFactory::getInstance();
    IBackendManager& mgr = bf.getBackendManager(config.backend);
    // backend.mutableMemoryManager().setMemoryLimit(config.maxMem_GB * 1e9); // TESTVALUE
    return mgr;
}


size_t StandardDeconvolutionStrategy::getMaxMemoryPerCube(
    size_t ioThreads,
    size_t workerThreads,
    const IBackendMemoryManager& backend,
    std::shared_ptr<DeconvolutionAlgorithm> algorithm
){
    size_t availableMemory = backend.getAvailableMemory();

    size_t memoryBuffer = 1e9;
    if (availableMemory < memoryBuffer) throw std::runtime_error("Available memory too low");
    availableMemory -= memoryBuffer;

    FFTWorkspaceCopiesEstimator estimator = [&backend](const CuboidShape& shape) {
        return backend.estimateFFTWorkspaceCopies(shape);
    };

    return computeMaxMemoryPerCube(
        availableMemory,
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






