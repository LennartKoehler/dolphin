#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include "UtlImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"
#include "backend/BackendFactory.h"
#include "backend/Exceptions.h"
#include "deconvolution/ImageMap.h"
#include "HelperClasses.h"



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
    std::unique_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(config);
    std::unique_ptr<IBackend> backend = getBackend(config);

    size_t t = config.nThreads;
    size_t memoryPerCube = maxMemoryPerCube(t, config.maxMem_GB * 1e9, algorithm.get());
    Padding cubePadding = getCubePadding(imageSize, psfs);
    RectangleShape idealCubeSizeUnpadded = getCubeShape(memoryPerCube, config.nThreads, imageSize, cubePadding);
    Padding imagePadding = getImagePadding(imageSize, idealCubeSizeUnpadded, cubePadding);

    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = splitImageHomogeneous(idealCubeSizeUnpadded, cubePadding, imageSize);
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());
    
    for (size_t i = 0; i < cubeCoordinatesWithPadding.size(); ++i) {
        StandardCubeTaskDescriptor descriptor;
        descriptor.taskId = static_cast<int>(i);
        descriptor.channelNumber = 0; // Default channel
        descriptor.paddedBox = cubeCoordinatesWithPadding[i];
        descriptor.psfs = psfPointers;
        descriptor.estimatedMemoryUsage = estimateMemoryUsage(idealCubeSizeUnpadded, algorithm.get());
        
        tasks.push_back(std::make_unique<StandardCubeTaskDescriptor>(descriptor));
    }
    
    size_t totalTasks = tasks.size();
    return ChannelPlan{
        std::move(backend),
        std::move(algorithm),
        ExecutionStrategy::PARALLEL,
        std::move(imagePadding),
        std::move(tasks),
        totalTasks
    };
}


std::unique_ptr<DeconvolutionAlgorithm> StandardDeconvolutionStrategy::getAlgorithm(const DeconvolutionConfig& config) {    
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    std::unique_ptr<DeconvolutionAlgorithm> algorithm = fact.createUnique(config);
    return algorithm; 
}

std::unique_ptr<IBackend> StandardDeconvolutionStrategy::getBackend(const DeconvolutionConfig& config){
    BackendFactory& bf = BackendFactory::getInstance();
    std::unique_ptr<IBackend> backend = bf.createUnique(config.backenddeconv);
    backend->mutableMemoryManager().setMemoryLimit(config.maxMem_GB * 1e9); 
    return backend;
}


size_t StandardDeconvolutionStrategy::maxMemoryPerCube(
    size_t maxNumberThreads, 
    size_t maxMemory,
    const DeconvolutionAlgorithm* algorithm){
    
    size_t algorithmMemoryMultiplier = algorithm->getMemoryMultiplier();
    size_t memoryBuffer = 1e9;
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
    size_t width = 128;
    size_t height = 256;
    size_t depth = 128;

    return RectangleShape(width, height, depth) - cubePadding.before - cubePadding.after;
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