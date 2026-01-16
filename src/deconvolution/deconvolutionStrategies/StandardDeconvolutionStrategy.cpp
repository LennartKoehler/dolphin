#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include <iostream>
#include <omp.h>
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"
#include "backend/BackendFactory.h"
#include "dolphinbackend/Exceptions.h"
#include "HelperClasses.h"
#include "frontend/SetupConfig.h"



ChannelPlan StandardDeconvolutionStrategy::createPlan(
    const ImageMetaData& metadata, 
    const std::vector<PSF>& psfs,
    const DeconvolutionConfig& config
) {
    std::vector<std::shared_ptr<PSF>> psfPointers;
    for (const auto& psf : psfs) {
        psfPointers.push_back(std::make_shared<PSF>(psf));
    }

    RectangleShape imageSize = RectangleShape{metadata.imageWidth, metadata.imageLength, metadata.slices};
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(config);
    std::shared_ptr<IBackend> backend = getBackend(config);

    size_t t = config.nThreads;
    size_t memoryPerCube = maxMemoryPerCube(t, config.maxMem_GB * 1e9, algorithm.get());
    Padding cubePadding = getCubePadding(imageSize, psfs);
    RectangleShape idealCubeSizeUnpadded = getCubeShape(memoryPerCube, config.nThreads, imageSize, cubePadding);
    Padding imagePadding = getImagePadding(imageSize, idealCubeSizeUnpadded, cubePadding);

    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = splitImageHomogeneous(idealCubeSizeUnpadded, cubePadding, imageSize);
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());
    
    for (size_t i = 0; i < cubeCoordinatesWithPadding.size(); ++i) {
        CubeTaskDescriptor descriptor;
        descriptor.algorithm = algorithm;
        descriptor.backend = backend;
        descriptor.taskId = static_cast<int>(i);
        descriptor.channelNumber = 0; // Default channel
        descriptor.paddedBox = cubeCoordinatesWithPadding[i];
        descriptor.psfs = psfPointers;
        descriptor.estimatedMemoryUsage = estimateMemoryUsage(idealCubeSizeUnpadded, algorithm.get());
        
        tasks.push_back(std::make_unique<CubeTaskDescriptor>(descriptor));
    }
    
    size_t totalTasks = tasks.size();
    return ChannelPlan{
        ExecutionStrategy::PARALLEL,
        std::move(imagePadding),
        std::move(tasks),
        totalTasks
    };
}


std::shared_ptr<DeconvolutionAlgorithm> StandardDeconvolutionStrategy::getAlgorithm(const DeconvolutionConfig& config) {    
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = fact.createShared(config);
    return algorithm; 
}

std::shared_ptr<IBackend> StandardDeconvolutionStrategy::getBackend(const DeconvolutionConfig& config){
    BackendFactory& bf = BackendFactory::getInstance();
    std::shared_ptr<IBackend> backend = bf.createShared(config.backenddeconv);
    backend->mutableMemoryManager().setMemoryLimit(config.maxMem_GB * 1e9); 
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
    return cubeSize.volume * algorithm->getMemoryMultiplier() * sizeof(complex);
}

RectangleShape StandardDeconvolutionStrategy::getCubeShape(
    size_t memoryPerCube,
    size_t numberThreads,
    const RectangleShape& imageOriginalShape,
    const Padding& cubePadding
){
    size_t width = 256;
    size_t height = 256; 
    size_t depth = 64;

    
    RectangleShape cubeSize = RectangleShape(width, height, depth) - cubePadding.before - cubePadding.after;
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