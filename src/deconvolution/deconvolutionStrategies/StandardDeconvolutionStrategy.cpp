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
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/Postprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/Exceptions.h"
#include "dolphin/HelperClasses.h"
#include "dolphin/frontend/SetupConfig.h"



DeconvolutionPlan StandardDeconvolutionStrategy::createPlan(
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
    RectangleShape imageSize = RectangleShape{metadata.imageWidth, metadata.imageLength, metadata.slices};
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(deconvConfig);
    spdlog::get("deconvolution")->info("Using the following deconvolution config");
    deconvConfig.printValues();

    std::shared_ptr<IBackend> backend = getBackend(setupConfig);


    std::vector<std::shared_ptr<TaskContext>> contexts = createContexts(backend, setupConfig);

    size_t nThreads = setupConfig.nThreads;
    size_t memoryPerCube = maxMemoryPerCube(nThreads, setupConfig.maxMem_GB * 1e9, algorithm.get());
    Padding cubePadding = getCubePadding(imageSize, psfs);
    RectangleShape idealCubeSizeUnpadded = getCubeShape(memoryPerCube, setupConfig.nThreads, setupConfig.cubeSize, imageSize, cubePadding);
    Padding imagePadding = getImagePadding(imageSize, idealCubeSizeUnpadded, cubePadding);

    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = splitImageHomogeneous(idealCubeSizeUnpadded, cubePadding, imageSize);
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());
    


    for (size_t i = 0; i < cubeCoordinatesWithPadding.size(); ++i) {

        std::shared_ptr<TaskContext> context = contexts[i % contexts.size()]; // cycle through contexts and assign the context to that task

        tasks.push_back(std::make_unique<CubeTaskDescriptor>(
            static_cast<int>(i),
            cubeCoordinatesWithPadding[i],
            algorithm,
            estimateMemoryUsage(cubeCoordinatesWithPadding[i].box.dimensions, algorithm.get()),
            psfPointers,
            reader,
            writer,
            context
        ));
    }
    
    size_t totalTasks = tasks.size();
    spdlog::get("deconvolution")
        ->info("Successfully created deconvolution plan with {} total cubes. Each cube has size (width x height x depth) ({}) which includes padding (padding before, padding after) ({}, {})",
        totalTasks, (idealCubeSizeUnpadded + imagePadding.before + imagePadding.after).print(), imagePadding.before.print(), imagePadding.after.print());
    return DeconvolutionPlan{
        std::move(imagePadding),
        std::move(tasks),
        totalTasks
    };
}

std::unique_ptr<PSFPreprocessor> StandardDeconvolutionStrategy::createPSFPreprocessor() const {
    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend>)> psfPreprocessFunction = [&](
        const RectangleShape shape,
        std::shared_ptr<PSF> inputPSF,
        std::shared_ptr<IBackend> backend
            ) -> ComplexData* {
                Preprocessor::padToShape(inputPSF->image, shape, PaddingType::ZERO);
                ComplexData h = Preprocessor::convertImageToComplexData(inputPSF->image);
                ComplexData h_device = backend->getMemoryManager().copyDataToDevice(h);
                backend->getDeconvManager().octantFourierShift(h_device);
                backend->getDeconvManager().forwardFFT(h_device, h_device);
                backend->sync();
                return new ComplexData(std::move(h_device));
            };
    std::unique_ptr<PSFPreprocessor> preprocessor = std::make_unique<PSFPreprocessor>();
    preprocessor->setPreprocessingFunction(psfPreprocessFunction);
    return std::move(preprocessor);
}

std::vector<std::shared_ptr<TaskContext>> StandardDeconvolutionStrategy::createContexts(
    std::shared_ptr<IBackend> backend,
    const SetupConfig& config) const {
        int numberDevices = backend->getNumberDevices();
        numberDevices = std::min(numberDevices, config.nDevices);
        numberDevices = numberDevices < 1 ? 1 : numberDevices;

        std::vector<std::shared_ptr<TaskContext>> contexts;
        
        for (int i = 0; i < numberDevices; i++){        
            std::shared_ptr<IBackend> prototypebackend = backend->onNewThread(backend);
            std::shared_ptr<TaskContext> context = std::make_shared<TaskContext>(prototypebackend, config.nWorkerThreads, config.nIOThreads);

            std::unique_ptr<PSFPreprocessor> preprocessor = createPSFPreprocessor(); // new psfpreprocessor for each context because the psfs live on device
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

std::shared_ptr<IBackend> StandardDeconvolutionStrategy::getBackend(const SetupConfig& config){
    BackendFactory& bf = BackendFactory::getInstance();
    std::shared_ptr<IBackend> backend = bf.createShared<IBackend>(config.backend);
    // backend->mutableMemoryManager().setMemoryLimit(config.maxMem_GB * 1e9); // TESTVALUE
    return backend;
}


size_t StandardDeconvolutionStrategy::maxMemoryPerCube(
    size_t maxNumberThreads, 
    size_t maxMemory,
    const DeconvolutionAlgorithm* algorithm){
    
    size_t algorithmMemoryMultiplier = algorithm->getMemoryMultiplier();
    size_t memoryBuffer = 1e9;
    maxNumberThreads = maxNumberThreads == 0 ? 1 : maxNumberThreads;
    size_t memoryPerThread = maxMemory / maxNumberThreads;
    size_t memoryPerCube = memoryPerThread / algorithmMemoryMultiplier;
    return memoryPerCube; 
}

size_t StandardDeconvolutionStrategy::estimateMemoryUsage(
    const RectangleShape& cubeSize,
    const DeconvolutionAlgorithm* algorithm
){
    return cubeSize.volume * algorithm->getMemoryMultiplier() * sizeof(complex_t);
}

RectangleShape StandardDeconvolutionStrategy::getCubeShape(
    size_t memoryPerCube,
    size_t numberThreads,
    const RectangleShape& configCubeSize,
    const RectangleShape& imageOriginalShape,
    const Padding& cubePadding
){    
    RectangleShape cubeSize = configCubeSize - cubePadding.before - cubePadding.after;
    cubeSize.clamp(imageOriginalShape);
    assert(cubeSize > RectangleShape(0,0,0));
    return cubeSize;
}

Padding StandardDeconvolutionStrategy::getImagePadding(
    const RectangleShape& imageSize,
    const RectangleShape& cubeSizeUnpadded,
    const Padding& cubePadding
){
    RectangleShape paddingBefore = cubePadding.before;
    RectangleShape paddingAfter;

    paddingAfter.width = std::max(cubePadding.after.width, cubeSizeUnpadded.width - imageSize.width + cubePadding.before.width);
    paddingAfter.height = std::max(cubePadding.after.height, cubeSizeUnpadded.height - imageSize.height + cubePadding.before.height);
    paddingAfter.depth = std::max(cubePadding.after.depth, cubeSizeUnpadded.depth - imageSize.depth + cubePadding.before.depth);
    return Padding{paddingBefore, paddingAfter};
}

Padding StandardDeconvolutionStrategy::getCubePadding(const RectangleShape& image, const std::vector<PSF> psfs){
    std::vector<RectangleShape> psfSizes;
    for (const auto& psf : psfs){
        psfSizes.push_back(psf.image.getShape());
    }
    
    RectangleShape maxPsfShape{0, 0, 0};
    
    for (const auto& psf : psfSizes) {
        maxPsfShape.width = std::max(maxPsfShape.width, psf.width);
        maxPsfShape.height = std::max(maxPsfShape.height, psf.height);
        maxPsfShape.depth = std::max(maxPsfShape.depth, psf.depth);
    }
    
    RectangleShape paddingbefore = RectangleShape(
        static_cast<int>(maxPsfShape.width / 2),
        static_cast<int>(maxPsfShape.height / 2),
        static_cast<int>(maxPsfShape.depth / 2)
    );
    paddingbefore = paddingbefore + 1;
    return Padding{paddingbefore, paddingbefore};
}

void StandardDeconvolutionStrategy::configure(const SetupConfig& setupConfig) {
    // Base configuration for standard strategy - no special setup needed
    // This method can be extended by subclasses for specific configuration requirements
}