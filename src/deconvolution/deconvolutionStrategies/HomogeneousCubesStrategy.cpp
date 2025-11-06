#include "deconvolution/deconvolutionStrategies/HomogeneousCubesStrategy.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>


void HomogeneousCubesStrategy::validateConfiguration(
    const std::vector<PSF>& psfs,
    const RectangleShape imageShape,
    const int channelNumber,
    const DeconvolutionConfig& config,
    const std::shared_ptr<IBackend> backend,
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm) const {
    
    if (psfs.empty()) {
        throw std::runtime_error("PSFs not set or empty in HomogeneousCubesStrategy");
    }
    if (!algorithm) {
        throw std::runtime_error("Algorithm not set in HomogeneousCubesStrategy");
    }
    if (!backend) {
        throw std::runtime_error("Backend not set in HomogeneousCubesStrategy");
    }

    // Validate PSF data integrity
    if (psfs[0].image.slices.empty()) {
        throw std::runtime_error("PSF image has no slices");
    }
    if (psfs[0].image.slices[0].cols <= 0 || psfs[0].image.slices[0].rows <= 0) {
        throw std::runtime_error("PSF image has invalid dimensions");
    }

    // Validate image shape
    if (imageShape.width <= 0 || imageShape.height <= 0 || imageShape.depth <= 0) {
        throw std::runtime_error("Image shape has invalid dimensions");
    }

    // Validate config values
    if (config.nThreads == 0) {
        throw std::runtime_error("Number of threads must be greater than 0");
    }
    if (config.maxMem_GB <= 0) {
        throw std::runtime_error("Maximum memory must be greater than 0");
    }

    // Validate channels
    if (channelNumber <= 0) {
        throw std::runtime_error("Number of channels must be greater than 0");
    }
}

ImageMap<std::vector<std::shared_ptr<PSF>>> HomogeneousCubesStrategy::getStrategy(
    const std::vector<PSF>& psfs,
    const RectangleShape imageShape,
    const int channelNumber,
    const DeconvolutionConfig& config,
    const std::shared_ptr<IBackend> backend,
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm){
    
    validateConfiguration(psfs, imageShape, channelNumber, config, backend, algorithm);

    // Get PSF size from the first PSF in the input
    RectangleShape psfSize = RectangleShape{psfs[0].image.slices[0].cols, psfs[0].image.slices[0].rows, psfs[0].image.slices.size()};
    
    // Use the imageShape parameter
    RectangleShape imageSize = imageShape;

    size_t memoryPerCube = getMemoryPerCube(config.nThreads, config.maxMem_GB * 1e9, algorithm);
    RectangleShape idealCubeSize = getCubeShape(memoryPerCube, config.nThreads, imageSize, psfSize);
    std::vector<BoxCoord> cubeCoordinates = splitImageHomogeneous(idealCubeSize, imageSize);

    return addPSFS(cubeCoordinates, psfs);
}





ImageMap<std::vector<std::shared_ptr<PSF>>> HomogeneousCubesStrategy::addPSFS(std::vector<BoxCoord>& coords, const std::vector<PSF>& psfs){
    ImageMap<std::vector<std::shared_ptr<PSF>>> result;

    for (auto& coordinate : coords){
        std::vector<std::shared_ptr<PSF>> psfPointers;
        for (const auto& psf : psfs){
            psfPointers.push_back(std::make_shared<PSF>(psf));
        }
        result.add(coordinate, std::move(psfPointers));
    }
    return result;
}

size_t HomogeneousCubesStrategy::getMemoryPerCube(
    size_t maxNumberThreads, 
    size_t maxMemory,
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm){
    
    size_t algorithmMemoryMultiplier = algorithm->getMemoryMultiplier(); // how many copies of a cube does each algorithm have
    size_t memoryBuffer = 1e9; // TESTVALUE
    size_t memoryPerThread = maxMemory / maxNumberThreads;
    size_t memoryPerCube = memoryPerThread / algorithmMemoryMultiplier;
    return memoryPerCube; 
}

RectangleShape HomogeneousCubesStrategy::getCubeShape(
    size_t memoryPerCube,
    size_t numberThreads,
    RectangleShape imageOriginalShape,
    RectangleShape padding

) {
    // this function determines the shape into which the input image is cut
    // current strategy is to only slice the largest dimension while leaving the smaller two dimensions the same shape
    // the constraints are that all threads should be used but it all needs to fit on the available memory
    // due to padding it is most optimal (smallest 3dshape) to have all dimensions the same size as this reduces the increase in volume caused by padding
    // but this is difficult as we want to have all threads have a similar workload aswell as reducing the overhead of each thread having to read/write more than once
    // ideally we have number of cubes (all dim same length) of equal size equal to number of threads
    // there are different strategies to split the original image but this is just what I went with
    // it is useful to keep all cubes the same dimensionality as the psfs then only need to be transformed once into that shape and the fftw plans can be reused


    size_t maxMemCubeVolume = memoryPerCube / sizeof(complex); // cut into pieces so that they still fit on memory


    RectangleShape subimageShape = imageOriginalShape;
    std::array<int*, 3> sortedDimensionsSubimage = subimageShape.getDimensionsAscending();
    size_t maxThreadcubeLargestDim = (*sortedDimensionsSubimage[2] + numberThreads -1) / numberThreads; // ceiling divide

    RectangleShape tempMemory = imageOriginalShape;
    std::array<int*, 3> sortedDimensionsMemory = tempMemory.getDimensionsAscending();
    size_t maxMemCubeLargestDim = maxMemCubeVolume / (*sortedDimensionsMemory[0] * *sortedDimensionsMemory[1]);

    *sortedDimensionsSubimage[2] = std::min(maxMemCubeLargestDim, maxThreadcubeLargestDim);
    assert(*sortedDimensionsSubimage[2] != 0 && "[ERROR] getCubeShape: not enough memory to fit a single slice of the image");


    subimageShape.updateVolume();
    return subimageShape;

    // TODO could also start halfing the other dimension until it fits
    // idea: always half the largest dimension until:
    //      number of cubes = number of threads && size of cubes fits on memory

    // size_t cubeVolume = std::min(memCubeVolume, threadCubeVolume);
    // double scaleFactor = std::cbrt( static_cast<double>(cubeVolume) / imageShapePadded.volume);
    // subimageShape = imageOriginalShape * scaleFactor;
    // subimageShape.clamp(imageOriginalShape);

    // cubeShapePadded = subimageShape + padding;

}

std::vector<BoxCoord> HomogeneousCubesStrategy::splitImageHomogeneous(
    const RectangleShape& subimageShape,
    const RectangleShape& imageOriginalShape 
){
    std::vector<BoxCoord> cubePositions;
    // Calculate number of cubes in each dimension
    int cubesInDepth = (imageOriginalShape.depth + subimageShape.depth - 1) / subimageShape.depth;
    int cubesInWidth = (imageOriginalShape.width + subimageShape.width - 1) / subimageShape.width;
    int cubesInHeight = (imageOriginalShape.height + subimageShape.height - 1) / subimageShape.height;
    
    // Calculate total number of cubes
    int totalCubes = cubesInDepth * cubesInWidth * cubesInHeight;
    cubePositions.reserve(totalCubes);

    assert(imageOriginalShape >= subimageShape &&  "[ERROR] subimage has to be smaller than image");   
    for (int d = 0; d < cubesInDepth; ++d) {
        for (int w = 0; w < cubesInWidth; ++w) {
            for (int h = 0; h < cubesInHeight; ++h) {
                
                // Calculate current position in original image coordinates
                RectangleShape currentPos(
                    w * subimageShape.width,
                    h * subimageShape.height,
                    d * subimageShape.depth
                );

                // Calculate remaining size for this cube
                RectangleShape remainingSize(
                    std::min(subimageShape.width, imageOriginalShape.width - w * subimageShape.width),
                    std::min(subimageShape.height, imageOriginalShape.height - h * subimageShape.height),
                    std::min(subimageShape.depth, imageOriginalShape.depth - d * subimageShape.depth)
                );

                // Skip if no remaining size (shouldn't happen with proper calculation)
                if (remainingSize.depth <= 0 || remainingSize.width <= 0 || remainingSize.height <= 0) {
                    continue;
                }

                // Determine actual cube positions - use overlap for boundary cubes
                RectangleShape actualPos = currentPos;
                
                // If this would be the last cube and doesn't fit completely, shift it back to create overlap
                if (remainingSize.depth < subimageShape.depth && remainingSize.depth > 0) {
                    actualPos.depth = currentPos.depth - (subimageShape.depth - remainingSize.depth);
                }
                if (remainingSize.width < subimageShape.width && remainingSize.width > 0) {
                    actualPos.width = currentPos.width - (subimageShape.width - remainingSize.width);
                }
                if (remainingSize.height < subimageShape.height && remainingSize.height > 0) {
                    actualPos.height = currentPos.height - (subimageShape.height - remainingSize.height);
                }
                BoxCoord cube;
                cube.x = actualPos.width;
                cube.y = actualPos.height;
                cube.z = actualPos.depth;
                cube.dimensions = subimageShape;
                cubePositions.push_back(std::move(cube));
            }
        }
    }

    return cubePositions;
}

