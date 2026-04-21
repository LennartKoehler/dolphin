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
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <expected>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/Postprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphinbackend/Exceptions.h"
#include "dolphin/HelperClasses.h"
#include "dolphin/SetupConfig.h"



Result<DeconvolutionPlan> StandardDeconvolutionStrategy::createPlan(
    std::shared_ptr<ImageReader> reader,
    std::shared_ptr<ImageWriter> writer,
    const std::vector<PSF>& psfs,
    const DeconvolutionConfig& deconvConfig,
    const SetupConfig& setupConfig
) {
    std::vector<std::shared_ptr<PSF>> psfPointers;
    for (const auto& psf : psfs) {
        psfPointers.push_back(std::make_shared<PSF>(psf));
    }

    ImageMetaData metadata = reader->getMetaData();
    CuboidShape imageSize = CuboidShape{metadata.imageWidth, metadata.imageLength, metadata.slices};
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(deconvConfig);
    spdlog::get("deconvolution")->info("Using the following deconvolution config");
    deconvConfig.printValues();

    IBackendManager& manager = getBackendManager(setupConfig);

    size_t totalThreads = setupConfig.nThreads;
    size_t ioThreads = setupConfig.nIOThreads;
    size_t workerThreads = setupConfig.nWorkerThreads;

    std::vector<std::shared_ptr<TaskContext>> contexts = createContexts(manager, setupConfig.nDevices, workerThreads, ioThreads, totalThreads);

    size_t maxMemoryPerCube = getMaxMemoryPerCube(
        ioThreads,
        workerThreads,
        manager,
        algorithm
    );
    size_t maxMemCubeVolume = maxMemoryPerCube / sizeof(real_t);

    Result<std::pair<Padding, CuboidShape>> cubePaddingResult = getCubePadding(psfs, setupConfig.cubePadding, imageSize);
    Padding cubePadding = std::move(cubePaddingResult.value.first);
    CuboidShape minShape = std::move(cubePaddingResult.value.second); // has to be at least as big as every psf


    if (!cubePaddingResult.success) {
        return Result<DeconvolutionPlan>(cubePaddingResult);
    }
    // Result<CuboidShape> idealCubeSize = getCubeShape(maxMemoryPerCube, setupConfig.cubeSize, imageSize, cubePadding, workerThreads);
    // if (!idealCubeSize.success) {
    //     return Result<DeconvolutionPlan>(idealCubeSize);
    // }

    if(cubePadding.before + cubePadding.after < deconvConfig.featheringRadius)
        spdlog::get("deconvolution")->warn("Feathering radius ({}) is smaller than padding (which is probably the size of the psf) ({}), which can cause artifacts",
            deconvConfig.featheringRadius, (cubePadding.before + cubePadding.after).print());

    // Padding imagePadding = getImagePadding(imageSize, idealCubeSize.value, cubePadding);


    Result<std::vector<BoxCoordWithPadding>> cubeCoordinatesWithPaddingResult = splitImageHomogeneous(cubePadding, imageSize, maxMemCubeVolume, workerThreads, deconvConfig.imagePaddingType, minShape);
    if (!cubeCoordinatesWithPaddingResult.success) {
        return Result<DeconvolutionPlan>(cubeCoordinatesWithPaddingResult);
    }
    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = cubeCoordinatesWithPaddingResult.value;

    BoxCoordWithPadding workShape = cubeCoordinatesWithPadding[0];

    if (workShape.padding.getTotalPadding() / workShape.box.dimensions> 3)
        spdlog::get("deconvolution")->warn("Low memory, padding takes up most of the compute block. Padding is: ({}); the subimage is: ({})",
                                           cubePadding.getTotalPadding().print(), workShape.box.dimensions.print());

    size_t memoryPerTask = estimateMemoryUsage(workShape.getBox().dimensions, algorithm.get(), setupConfig);

    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());

    for (size_t i = 0; i < cubeCoordinatesWithPadding.size(); ++i) {

        std::shared_ptr<TaskContext> context = contexts[i % contexts.size()]; // cycle through contexts and assign the context to that task
        // TODO if the contexts are uneven in their compute capability then this would have to be more dynamic,
        // one idea would be not to assign the context here at all, and just attach the contexts to the deconvplan, not the individiaul tasks
        // then the deconvexecutor could dynamically attach contexts to tasks when that context is running out of tasks,
        // so like having a queue and only filling each context queue when that queue is getting empty, then slow queues wouldnt get that much work

        tasks.push_back(std::make_unique<CubeTaskDescriptor>(
            static_cast<int>(i),
            cubeCoordinatesWithPadding[i],
            algorithm,
            memoryPerTask,
            psfPointers,
            reader,
            writer,
            context
        ));
    }

    size_t totalTasks = tasks.size();
    assert (totalTasks > 0 && "No tasks in deconvolution plan");

    spdlog::get("deconvolution")
        ->info("Successfully created deconvolution plan with {} total cubes. Each cube has size (width x height x depth) ({}) which includes padding (padding before, padding after) ({}, {})",
        totalTasks, (workShape.getBox().dimensions).print(), cubePadding.before.print(), cubePadding.after.print());

    DeconvolutionPlan plan {
        std::move(tasks),
        totalTasks
    };
    return std::move(Result<DeconvolutionPlan>::ok(std::move(plan)));
}



std::unique_ptr<PSFPreprocessor> StandardDeconvolutionStrategy::createPSFPreprocessor() const {

    std::function<std::unique_ptr<ComplexData>(const CuboidShape, std::shared_ptr<PSF>, IBackend&)> psfPreprocessFunction = [&](
        const CuboidShape targetShape,
        std::shared_ptr<PSF> inputPSF,
        IBackend& backend
    ) -> std::unique_ptr<ComplexData> {
            Preprocessor::padToShape(inputPSF->image, targetShape, PaddingType::ZERO);
            RealData h = Preprocessor::convertImageToRealData(inputPSF->image);
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

std::vector<std::shared_ptr<TaskContext>> StandardDeconvolutionStrategy::createContexts(
    IBackendManager& manager,
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

            std::unique_ptr<PSFPreprocessor> preprocessor = createPSFPreprocessor(); // new psfpreprocessor for each context because the psfs live on devicecudabac
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
    IBackendManager& manager,
    std::shared_ptr<DeconvolutionAlgorithm> algorithm
){
    BackendConfig backendConfig; //TODO TESTVALUE
    IBackendMemoryManager& backend = manager.getBackendMemoryManager(backendConfig);

    size_t availableMemory = backend.getAvailableMemory();

    size_t memoryBuffer = 1e9;
    if (availableMemory < memoryBuffer) throw std::runtime_error("Available memory too low");
    availableMemory -= memoryBuffer;

    int ioCopies = 3; //image, psf, result, but psf only allocated once in total
    size_t ioAllocations = ioThreads * ioCopies;
    size_t workerAllocations = workerThreads * algorithm->getMemoryMultiplier();

    size_t threadallocations = ioAllocations + workerAllocations;
    assert(threadallocations != 0 && "Error, no threadallocations");

    size_t memoryPerCube = availableMemory / threadallocations ;

    return memoryPerCube;
}

size_t StandardDeconvolutionStrategy::estimateMemoryUsage(
    const CuboidShape& cubeSize,
    const DeconvolutionAlgorithm* algorithm,
    const SetupConfig& config
){
    int ioCopies = 3; //image, psf, result
    size_t ioAllocations = config.nIOThreads * cubeSize.getVolume() * ioCopies * sizeof(complex_t);
    size_t workerAllocations = config.nWorkerThreads * cubeSize.getVolume() * algorithm->getMemoryMultiplier() * sizeof(complex_t);
    return ioAllocations + workerAllocations;
}

// Result<CuboidShape> StandardDeconvolutionStrategy::getCubeShape(
//     size_t maxMemoryPerCube,
//     const CuboidShape& configCubeSize,
//     const CuboidShape& imageOriginalShape,
//     const Padding& cubePadding,
//     size_t nWorkerThreads
// ){
//     CuboidShape cubeSize;
//     if (configCubeSize.getVolume() != 0){
//         cubeSize = configCubeSize;
//     }
//     else{
//         size_t maxMemCubeVolume = maxMemoryPerCube / sizeof(complex_t); // cut into pieces so that they still fit on memory
//
//         size_t ncubes = 1;
//         cubeSize = imageOriginalShape;
//
//         // cubeSize.toNextPowerOfTwo();
//
//         size_t volume = (cubeSize + cubePadding.before + cubePadding.after).getVolume();
//         std::array<int*, 3> tempCubeAccessor  = cubeSize.getReference();
//         int dimIterator = 2;
//
//         while (volume > maxMemCubeVolume || ncubes <= nWorkerThreads){
//             dimIterator = (++dimIterator) % 3;
//             *tempCubeAccessor[dimIterator] -= 10; //always only reduce one dimension
//             volume = (cubeSize + cubePadding.before + cubePadding.after).getVolume();
//             ncubes = imageOriginalShape.getNumberSubcubes(cubeSize);
//         }
//         // cubeSize = cubeSize / 2; //TESTVALUE
//         cubeSize = cubeSize + cubePadding.before + cubePadding.after;
//
//         // cubeSize.toNextPowerOfTwo();
//     }
//
//
//     // Apply to the padded cube dimensions (total buffer size)
//     cubeSize.width = nextSmooth(cubeSize.width);
//     cubeSize.height = nextSmooth(cubeSize.height);
//     cubeSize.depth = nextSmooth(cubeSize.depth);
//     // ---------------------------------------------
//
//     if (cubeSize < cubePadding.before + cubePadding.after)
//     {
//         return Result<CuboidShape>::fail(
//             "Cube has invalid shape as it needs to be larger than padding, cubeSize: " + cubeSize.print() + " padding: " + cubePadding.before.print() + cubePadding.after.print());
//     }
//
//     return Result<CuboidShape>::ok(std::move(cubeSize));
// }

int StandardDeconvolutionStrategy::getNextPowerOfTwo(int v) const {
    int p = 2;
    while (p < v){
        p <<= 1;
    }
    return p;
}

// unused, its intrinsic in cube padding and imagesplit
// the tiffreader then understands it needs "negative" padding
// Padding StandardDeconvolutionStrategy::getImagePadding(
//     const CuboidShape& imageSize,
//     const CuboidShape& cubeSizeUnpadded,
//     const Padding& cubePadding
// ){
//     CuboidShape paddingBefore = cubePadding.before;
//     CuboidShape paddingAfter;
//
//     // if image is smaller than one single cubeSizeUnpadded, then we pad after
//     paddingAfter.width = std::max(cubePadding.after.width, cubeSizeUnpadded.width - imageSize.width + cubePadding.before.width);
//     paddingAfter.height = std::max(cubePadding.after.height, cubeSizeUnpadded.height - imageSize.height + cubePadding.before.height);
//     paddingAfter.depth = std::max(cubePadding.after.depth, cubeSizeUnpadded.depth - imageSize.depth + cubePadding.before.depth);
//     return Padding{paddingBefore, paddingAfter};
// }

std::vector<CuboidShape> StandardDeconvolutionStrategy::getPSFSizes(const std::vector<PSF>& psfs){
    std::vector<CuboidShape> psfSizes;
    for (const auto& psf : psfs){
        psfSizes.push_back(psf.image.getShape());
    }
    return psfSizes;
}



// while were at it just get the minSize the subimage has to be (so that its at least as big as every psf)
Result<std::pair<Padding, CuboidShape>> StandardDeconvolutionStrategy::getCubePadding(const std::vector<PSF>& psfs, const CuboidShape& configPadding, const CuboidShape& imageSize){
    Padding padding;
    std::vector<CuboidShape> psfSizes = getPSFSizes(psfs);

    CuboidShape largestPSF = getLargestShape(psfSizes);

    if (configPadding.getVolume() == -1){
        DefaultPaddingStrategy stratd{};
        stratd.init(configPadding);
        padding = stratd.getPadding(psfSizes);
    }
    else{
        ManualPaddingStrategy stratm{};
        stratm.init(configPadding);
        padding = stratm.getPadding(psfSizes);
        if (padding.getTotalPadding() + imageSize < largestPSF)
        {
            return Result<Padding>::fail(
                "Image with custom padding smaller than PSF");
        }
    }

    if (padding.before < CuboidShape{0,0,0} ||
        padding.after  < CuboidShape{0,0,0})
    {
        return Result<Padding>::fail(
            "Padding for cubes is smaller than zero");
    }

    assert(padding.getTotalPadding() + imageSize >= largestPSF);

    return Result<std::pair<Padding, CuboidShape>>::ok(std::pair<Padding, CuboidShape>(std::move(padding), std::move(padding.getTotalPadding() + CuboidShape{1,1,1})));
}


void StandardDeconvolutionStrategy::configure(const SetupConfig& setupConfig) {
    // Base configuration for standard strategy - no special setup needed
    // This method can be extended by subclasses for specific configuration requirements
}
