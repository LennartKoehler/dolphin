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
#include "dolphinbackend/MemoryFootprint.h"
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin_image/HelperClasses.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/PSFCreator.h"
#include "dolphin/psf/PSFGeneratorFactory.h"



Result<DeconvolutionPlan> StandardDeconvolutionStrategy::createPlan(
    std::shared_ptr<ImageReader> reader,
    std::shared_ptr<ImageWriter> writer,
    PSFHandler& psfHandler,
    const DeconvolutionConfig& deconvConfig,
    const SetupConfig& setupConfig
) {

    IBackendManager& backendManager = BackendFactory::getBackendManagerStatic(setupConfig.backend);

    // these are necessary as the cpu can have actual seperate threads for each task or use multiple threads for one task
    BackendConfig workerConfig; // these are configured by the backend so that when its given a task it know what to do
    BackendConfig ioConfig;

    auto [resolvedThreads, nDevices] = resolveThreadsAndDevices(
        backendManager,
        setupConfig.nDevices,
        setupConfig.threadConfig,
        workerConfig,
        ioConfig);

    size_t ioThreads = resolvedThreads.nIOThreads;
    size_t workerThreads = resolvedThreads.nWorkerThreads;
    size_t nReaderThreads = resolvedThreads.resolveReaderThreads();

    // Resolve the memory budgets before anything is allocated: explicit user
    // limits (maxMemHost_gb / maxMemDevice_gb) clamped down to what is
    // available, or ~90% of it when unset. These caps are enforced at runtime
    // (see applyBackendMemoryLimits) and drive the cube size.
    MemoryBudget budget = resolveMemoryBudget(setupConfig, backendManager);

    // set up writer
    WriterCompressionConfig writerCompressionConfig;
    writerCompressionConfig.compressionScheme = static_cast<uint16_t>(setupConfig.outputCompression);
    writerCompressionConfig.compressionLevel = setupConfig.outputCompressionLevel;

    WriterConfig writerConfig;
    writerConfig.tileWidth = static_cast<uint32_t>(setupConfig.tileWidth);
    writerConfig.tileLength = static_cast<uint32_t>(setupConfig.tileLength);

    writer->configure(writerCompressionConfig, writerConfig);
    // -----------------------

    ImageMetaData metadata = reader->getMetaData();
    // this cubepadding might still change due to good shapes for DFT. But this is the minimum!
    CuboidShape imageSize = CuboidShape{metadata.imageWidth, metadata.imageLength, metadata.slices};

    // PSF handling
    psfHandler.generatePSFs(setupConfig, deconvConfig, imageSize);
    Result<PaddingScheme> paddingResult = psfHandler.getPadding(deconvConfig);
    if (!paddingResult.success) {
        throw std::runtime_error("Error while getting Padding");
    }
    PaddingScheme paddingScheme = std::move(paddingResult.value);
    assert(paddingScheme.insidePadding.before >= paddingScheme.imagePadding.before);

    std::shared_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(deconvConfig);

    spdlog::get("deconvolution")->debug("Using the following deconvolution config");
    deconvConfig.printValues();

    spdlog::get("deconvolution")->debug("Using the following setup config");
    setupConfig.printValues();




    const size_t nPSFs = psfHandler.getPSFCount();

    MemoryComponent& readerComponent = reader->getMemoryComponent(nReaderThreads);
    MemoryComponent& writerComponent = writer->getMemoryComponent();
    // Each manager reports and caps the plane it allocates on (CPU: host,
    // CUDA: device) via its own memory component.
    MemoryComponent& backendComponent = backendManager.getMemoryComponent();
    MemoryComponent& psfComponent = psfHandler.getMemoryComponent(nPSFs, nDevices);

    // Memory footprint: every memory-owning system reports its own demand for
    // a candidate cube shape; the splitter compares it against the budget.
    MemoryFootprint footprint = buildMemoryFootprint(
        readerComponent,
        writerComponent,
        backendComponent,
        psfComponent,
        *algorithm,
        paddingScheme,
        ioThreads,
        workerThreads,
        !budget.deviceIsHost);

    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = getCubes(
        budget,
        footprint,
        workerThreads * nDevices,
        deconvConfig,
        imageSize,
        paddingScheme);


    std::vector<std::shared_ptr<TaskContext>> contexts = createContexts(
        backendManager,
        psfHandler,
        nDevices,
        workerThreads,
        ioThreads,
        ioConfig,
        workerConfig);

    BoxCoordWithPadding workShape = cubeCoordinatesWithPadding[0];


    if (workShape.padding.getTotalPadding() / workShape.box.dimensions> 3)
        spdlog::get("deconvolution")->warn("Low memory, padding takes up most of the compute block. Padding is: ({}); the subimage is: ({})",
                                           workShape.padding.getTotalPadding().print(), workShape.box.dimensions.print());

    // Derive the runtime caps from the same footprint that sized the cubes:
    // the reader gets what it reported for the final shape (enforced with
    // backpressure), the writer's peak is reserved (it has no cap mechanism),
    // and the backend gets the rest of the host pool and the device pool
    // (enforced with a clear error).
    const CuboidShape paddedShape = workShape.getPaddedShape();
    MemoryRequirement cubeNeed = footprint.total(paddedShape);
    budget.readerCache_byte = readerComponent.footprint(paddedShape).hostBytes;
    budget.writerReserve_byte = writerComponent.footprint(paddedShape).hostBytes;
    budget.backendHost_byte = budget.hostTotal_byte - budget.readerCache_byte - budget.writerReserve_byte;

    // set up reader and writer handlers; the reader's cache cap is applied
    // through its memory component (facade)
    ReaderConfig readerConfig;
    readerConfig.numReaderThreads = nReaderThreads;
    int readerChannel = 0; //TODO unused
    reader->configure(readerChannel, readerConfig);
    readerComponent.setMemoryLimit(MemoryRequirement{budget.readerCache_byte, 0});
    std::shared_ptr<ReaderHandler> readerHandler = std::make_shared<ReaderHandler>(reader, deconvConfig.paddingFillType);
    std::shared_ptr<WriterHandler> writerHandler = std::make_shared<WriterHandler>(writer);

    applyBackendMemoryLimits(backendComponent, budget);

    // psfHandler.fitPSFsToShape(paddedShape); // the psfpreprocessor does this
    std::vector<std::shared_ptr<PSF>>& psfs = psfHandler.getPSFs(paddedShape);

    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());

    std::shared_ptr<CubeTaskSharedDescriptor> sharedTaskDescriptor = std::make_unique<CubeTaskSharedDescriptor>(
            algorithm,
            cubeNeed,
            psfs,
            readerHandler,
            writerHandler);

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

    spdlog::get("deconvolution")
        ->debug("Memory footprint per cube ({}): {}", paddedShape.print(), footprint.breakdown(paddedShape));

    spdlog::get("deconvolution")
        ->debug("Memory budgets: host {:.2f} GiB (reader cache {:.2f} GiB, writer reserve {:.2f} GiB, backend allocations {:.2f} GiB), device {:.2f} GiB per device ({} device(s)); estimated demand per cube: {:.2f} MiB host, {:.2f} MiB device",
        budget.hostTotal_byte / 1073741824.0, budget.readerCache_byte / 1073741824.0, budget.writerReserve_byte / 1073741824.0, budget.backendHost_byte / 1073741824.0,
        budget.deviceTotal_byte / 1073741824.0, nDevices,
        cubeNeed.hostBytes / 1048576.0, cubeNeed.deviceBytes / 1048576.0);

    DeconvolutionPlan plan {
        std::move(tasks),
        totalTasks
    };
    return std::move(Result<DeconvolutionPlan>::ok(std::move(plan)));
}


MemoryFootprint StandardDeconvolutionStrategy::buildMemoryFootprint(
    const MemoryComponent& readerComponent,
    const MemoryComponent& writerComponent,
    const MemoryComponent& backendComponent,
    const MemoryComponent& psfComponent,
    const DeconvolutionAlgorithm& algorithm,
    const PaddingScheme& paddingScheme,
    size_t ioThreads,
    size_t workerThreads,
    bool algorithmOnDevice) {
    // The strategy-owned terms have no owner class: the footprint creates and
    // keeps them alive so the result can be returned by value. Each component
    // reports ONE instance; the count is the number of concurrent instances
    // of that system (the scheduling lives here).
    MemoryFootprint footprint;
    footprint.add(readerComponent);
    footprint.add(writerComponent);
    footprint.add(std::make_unique<const IOStagingMemoryComponent>(), ioThreads);
    footprint.add(std::make_unique<const AlgorithmWorkingSetMemoryComponent>(algorithm, algorithmOnDevice), workerThreads);
    footprint.add(backendComponent);
    footprint.add(psfComponent);
    // Padding runs per io thread: each concurrently filling thread holds its
    // unpadded source cube next to the staging buffer it fills.
    footprint.add(std::make_unique<const PaddingPeakMemoryComponent>(paddingScheme.insidePadding), ioThreads);
    return footprint;
}


std::vector<BoxCoordWithPadding> StandardDeconvolutionStrategy::getCubes(
    const MemoryBudget& budget,
    const MemoryFootprint& footprint,
    const size_t& minNumberCubes,
    const DeconvolutionConfig& deconvConfig,
    const CuboidShape& imageSize,
    const PaddingScheme& paddingScheme
) const {

    if(paddingScheme.insidePadding.before + paddingScheme.insidePadding.after < deconvConfig.featheringRadius)
        spdlog::get("deconvolution")->warn("Feathering radius ({}) is smaller than padding (which is probably the size of the psf) ({}), which can cause artifacts",
            deconvConfig.featheringRadius, (paddingScheme.insidePadding.before + paddingScheme.insidePadding.after).print());

    CuboidShape minShape = paddingScheme.insidePadding.getTotalPadding() + 1;

    if (!budget.fits(footprint.total(minShape))){
        throw std::runtime_error("Deconvolution with the largest PSF and padding requires too much memory. The minimum size for one cube would be: "
            + minShape.print() + " which needs: " + footprint.breakdown(minShape));
    }
    Result<std::vector<BoxCoordWithPadding>> cubeCoordinatesWithPaddingResult;
    if (deconvConfig.cubeSize[0] > 0 && deconvConfig.cubeSize[1] > 0 && deconvConfig.cubeSize[2] > 0){
        // Fixed cube size requested: still verify it fits into the memory
        // budget instead of silently running out of memory later.
        CuboidShape fixedPadded = CuboidShape(deconvConfig.cubeSize)
            + paddingScheme.insidePadding.before + paddingScheme.insidePadding.after;
        if (!budget.fits(footprint.total(fixedPadded))) {
            throw std::runtime_error(
                "Configured cubeSize " + fixedPadded.print() + " (with padding) does not fit into the memory limits"
                " (max_mem_host_gb / max_mem_device_gb). It needs: " + footprint.breakdown(fixedPadded)
                + " Reduce cubeSize or raise the limits.");
        }
        cubeCoordinatesWithPaddingResult = splitImageHomogeneousFixedSize(
            paddingScheme.insidePadding,
            paddingScheme.imagePadding,
            imageSize,
            CuboidShape(deconvConfig.cubeSize));
    }
    else {
        cubeCoordinatesWithPaddingResult = splitImageHomogeneous(
            paddingScheme.insidePadding,
            paddingScheme.imagePadding,
            imageSize,
            budget,
            footprint,
            minNumberCubes,
            minShape,
            deconvConfig.sliceAllDimensions);
    }

    if (!cubeCoordinatesWithPaddingResult.success) {
throw std::runtime_error("Error while splitting image: " + cubeCoordinatesWithPaddingResult.getErrorString());
    }
    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = cubeCoordinatesWithPaddingResult.value;
    return cubeCoordinatesWithPadding;
}


std::pair<ThreadConfig, size_t> StandardDeconvolutionStrategy::resolveThreadsAndDevices(
    IBackendManager& manager,
    const size_t& nDevices,
    const ThreadConfig& threadConfig,
    BackendConfig& workerconfig,
    BackendConfig& ioconfig
) const{
    ThreadConfig resolvedConfig = threadConfig;

    size_t numberDevices = nDevices == 0 ? manager.getNumberDevices() : std::min(nDevices, manager.getNumberDevices());

    // The backend owns the thread configuration: it resolves the user's intent
    // (Auto/Total/Full) into concrete executor pool sizes (written back into
    // resolvedConfig) and fills ioconfig/workerconfig with the per-op
    // parallelism (e.g. OMP threads per worker). The strategy just reads the
    // resolved values back.
    manager.setThreadDistribution(resolvedConfig, ioconfig, workerconfig);

    return std::pair<ThreadConfig, size_t>{resolvedConfig, numberDevices};
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



size_t StandardDeconvolutionStrategy::resolveMemoryLimit(float limit_gb, size_t available_byte) {
    // -1 (default) and 0 both mean "auto": use almost all currently available
    // memory, keeping a safety margin for fragmentation, other processes and
    // model inaccuracies.
    if (limit_gb <= 0.0f) {
        return static_cast<size_t>(static_cast<double>(available_byte) * AUTO_MEMORY_SAFETY_FACTOR);
    }
    // Explicit limit: a hard upper bound, clamped down to what actually exists.
    const size_t limit_byte = static_cast<size_t>(limit_gb * 1024.0 * 1024.0 * 1024.0);
    return std::min(limit_byte, available_byte);
}

MemoryBudget StandardDeconvolutionStrategy::resolveMemoryBudget(
    const SetupConfig& config,
    IBackendManager& backendManager
) const {
    MemoryBudget budget;
    budget.deviceIsHost = (config.backend == HOST_BACKEND);

    // Host pool: reader cache + io staging (+ everything for the CPU backend).
    // Queried through the host backend manager; constructs no backends.
    const size_t availableHost = BackendFactory::getBackendManagerStatic(HOST_BACKEND).getAvailableMemoryPerDevice();
    budget.hostTotal_byte = resolveMemoryLimit(config.maxMemHost_gb, availableHost);

    if (budget.deviceIsHost) {
        budget.deviceTotal_byte = budget.hostTotal_byte;
    } else {
        // Device pool per accelerator: queried without constructing backends.
        const size_t availableDevice = backendManager.getAvailableMemoryPerDevice();
        budget.deviceTotal_byte = resolveMemoryLimit(config.maxMemDevice_gb, availableDevice);
    }
    return budget;
}

void StandardDeconvolutionStrategy::applyBackendMemoryLimits(
    MemoryComponent& backendComponent,
    const MemoryBudget& budget
) const {
    // Both caps are applied through the managers' memory components (facades),
    // so the strategy does not touch the managers' memory APIs directly.
    if (budget.deviceIsHost) {
        // CPU: one shared tracker covers io staging and all "device"
        // allocations (host and compute backends share it).
        backendComponent.setMemoryLimit(MemoryRequirement{budget.backendHost_byte, 0});
    } else {
        // Host-side staging runs through the host backend manager; the compute
        // backend (already resolved by the caller) gets the device cap. The
        // Host-side staging runs through the host backend manager; the compute
        // backend (already resolved by the caller) gets the device cap. The
        // host manager is always the CPU manager, which caps the host plane.
        BackendFactory::getBackendManagerStatic(HOST_BACKEND).getMemoryComponent()
            .setMemoryLimit(MemoryRequirement{budget.backendHost_byte, 0});
        backendComponent.setMemoryLimit(MemoryRequirement{0, budget.deviceTotal_byte});
    }
}
