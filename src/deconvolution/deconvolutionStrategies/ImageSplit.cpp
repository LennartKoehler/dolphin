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

#include "dolphin/HelperClasses.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphinbackend/CuboidShape.h"
#include <stdexcept>

// --- Optimize for FFTW (Smooth Numbers) ---
// A "smooth" number has prime factors of only 2, 3, and 5.
// This ensures FFTW can use its fastest algorithms.
static bool isSmooth(int n){
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
};

static int nextSmooth(int dim){
    if (dim <= 0) return dim;
    while (!isSmooth(dim)) {
        dim++;
    }
    return dim;
};
static int previousSmooth(int dim){
    if (dim <= 0) return dim;
    dim--;
    while (!isSmooth(dim)) {
        dim--;
    }
    return dim;
};

void adjustDimensionsEdgeConditions(
    BoxCoordWithPadding& currentCube,
    const CuboidShape& imageOriginalShape,
    const CuboidShape& remainingSize,
    const PaddingType& imagePadding){

    // no cube padding for first and last cube, only padding between cubes but not outside
    // this way the total image volume isnt increased which makes it much faster
    // the outside padding would be artificial padding anyway, like mirror padding
    // but not padding at all is like periodic boundaries as the fourier transform is cyclic
    if (imagePadding == PaddingType::NONE){
        if (currentCube.box.position.depth == 0) {
            currentCube.box.dimensions.depth += currentCube.padding.before.depth;
            currentCube.padding.before.depth = 0;
        }
        if (currentCube.box.position.width == 0) {
            currentCube.box.dimensions.width += currentCube.padding.before.width;
            currentCube.padding.before.width = 0;
        }
        if (currentCube.box.position.height == 0) {
            currentCube.box.dimensions.height += currentCube.padding.before.height;
            currentCube.padding.before.height = 0;
        }
        // if the subimage + padding after is larger or equal to original image, then no padding
        if (currentCube.box.dimensions.width + currentCube.padding.after.width >= imageOriginalShape.width) {
            currentCube.box.dimensions.width += currentCube.padding.after.width;
            currentCube.padding.after.width = 0;
        }
        if (currentCube.box.dimensions.height + currentCube.padding.after.height >= imageOriginalShape.height) {
            currentCube.box.dimensions.height += currentCube.padding.after.height;
            currentCube.padding.after.height = 0;
        }
        if (currentCube.box.dimensions.depth + currentCube.padding.after.depth >= imageOriginalShape.depth) {
            currentCube.box.dimensions.depth += currentCube.padding.after.depth;
            currentCube.padding.after.depth = 0;
        }
    }
    // If subimage cube  is larger than image in any dimension, adjust to make padding after larger while making dimensions of box smaller
    // so basically if psf is larger than image in any dimension
    if (currentCube.box.dimensions.width > imageOriginalShape.width) {
        currentCube.padding.after.width = currentCube.padding.after.width + currentCube.box.dimensions.width - imageOriginalShape.width;
        currentCube.box.dimensions.width = imageOriginalShape.width;
    }
    if (currentCube.box.dimensions.height > imageOriginalShape.height) {
        currentCube.padding.after.height = currentCube.padding.after.height + currentCube.box.dimensions.height - imageOriginalShape.height;
        currentCube.box.dimensions.height = imageOriginalShape.height;
    }
    if (currentCube.box.dimensions.depth > imageOriginalShape.depth) {
        currentCube.padding.after.depth = currentCube.padding.after.depth + currentCube.box.dimensions.depth - imageOriginalShape.depth;
        currentCube.box.dimensions.depth = imageOriginalShape.depth;
    }

    // If this would be the last cube and doesn't fit completely, shift it back to create overlap with the previous cube and therefore prevent going out of bounds
    if (remainingSize.depth < currentCube.box.dimensions.depth && remainingSize.depth > 0) {
        currentCube.box.position.depth -= (currentCube.box.dimensions.depth - remainingSize.depth);
    }
    if (remainingSize.width < currentCube.box.dimensions.width && remainingSize.width > 0) {
        currentCube.box.position.width -= (currentCube.box.dimensions.width - remainingSize.width);
    }
    if (remainingSize.height < currentCube.box.dimensions.height && remainingSize.height > 0) {
        currentCube.box.position.height -= (currentCube.box.dimensions.height - remainingSize.height);
    }
}

// add new cube recursively
void addCubeRecursion(
    std::vector<BoxCoordWithPadding>& cubePositions,
    BoxCoordWithPadding& currentCube,
    const CuboidShape& imageOriginalShape,
    const PaddingType& imagePadding){

    assert(currentCube.box.dimensions.getVolume() > 0);

    // next row
    if (currentCube.box.position.width >= imageOriginalShape.width){
        currentCube.box.position.width = 0;
        currentCube.box.position.height += currentCube.box.dimensions.height;
        addCubeRecursion(
            cubePositions,
            currentCube,
            imageOriginalShape,
            imagePadding);
        return;
    }
    // next slice
    if (currentCube.box.position.height >= imageOriginalShape.height){
        currentCube.box.position.height = 0;
        currentCube.box.position.depth += currentCube.box.dimensions.depth;
        addCubeRecursion(
            cubePositions,
            currentCube,
            imageOriginalShape,
            imagePadding);
        return;
    }
    // were done
    if (currentCube.box.position.depth >= imageOriginalShape.depth)
        return;

    CuboidShape remainingSize = imageOriginalShape - currentCube.box.position;

    adjustDimensionsEdgeConditions(
        currentCube,
        imageOriginalShape,
        remainingSize,
        imagePadding);

    cubePositions.push_back(currentCube);


    // next cube (column)
    currentCube.box.position.width += currentCube.box.dimensions.width;
    addCubeRecursion(
        cubePositions,
        currentCube,
        imageOriginalShape,
        imagePadding);
}


bool decreaseSize(std::array<int*, 3>& tempCubeAccessor, int& dimIterator, const CuboidShape& minSize){
    int tries = 0;
    while (tries < 3){
        tries++;
        dimIterator = (++dimIterator) % 3;
        int newSize = previousSmooth(*tempCubeAccessor[dimIterator]);
        if (newSize >= minSize.getArray()[dimIterator]){
            *(tempCubeAccessor[dimIterator]) = newSize;
            return true;
        }
    }
    return false;
}

// since there are so many competing conditions for the cubes like maxSize (bc of memory), min number cubes(to e.g. use all devices)
// but also keep the size a smooth number for fftw, and somewhat "dynamic" padding if at an edge or not etc.
// so this is more of a just try out a bunch and when all conditions are sufficiently met then keep that plan
// i assume one could have also had a more complicated "model" of all the interactions and get a cube distribution that way
Result<std::vector<BoxCoordWithPadding>> splitImageHomogeneous(
    const Padding& cubePadding,
    const CuboidShape& imageOriginalShape,
    const size_t& maxVolumePerCube,
    const size_t& minNumberCubes,
    const PaddingType& imagePadding,
    const CuboidShape& minSize)
    {
    assert(minSize > cubePadding.getTotalPadding());


    CuboidShape currentMaxSize;

    if (imagePadding == PaddingType::NONE) currentMaxSize = imageOriginalShape;
    else currentMaxSize = imageOriginalShape + cubePadding.before + cubePadding.after;

    currentMaxSize.setMin(minSize); // because it has to be atleast as big as the psf

    // get next smooth size for faster fftw
    currentMaxSize.width = nextSmooth(currentMaxSize.width);
    currentMaxSize.height = nextSmooth(currentMaxSize.height);
    currentMaxSize.depth = nextSmooth(currentMaxSize.depth);

    std::array<int*, 3> tempCubeAccessor  = currentMaxSize.getReference();
    int dimIterator = 2;

    std::vector<BoxCoordWithPadding> cubePositions;
    int ncubes = 0;
    CuboidShape cubeSizeToUse = currentMaxSize;

    while (ncubes < minNumberCubes){

        cubePositions.clear(); // reset

        cubeSizeToUse = currentMaxSize - cubePadding.before - cubePadding.after;

        BoxCoordWithPadding startCube{BoxCoord{CuboidShape(0,0,0), cubeSizeToUse}, cubePadding};

        if (startCube.getBox().dimensions.getVolume() < maxVolumePerCube){

            addCubeRecursion(
                cubePositions,
                startCube,
                imageOriginalShape,
                imagePadding);
            ncubes = cubePositions.size();
        }

        bool success = decreaseSize(tempCubeAccessor, dimIterator, minSize);
        if (!success){
            return Result<std::vector<BoxCoordWithPadding>>::fail("Not enough memory to fit the smallest possible cube");
        }
    }
    return Result<std::vector<BoxCoordWithPadding>>::ok(std::move(cubePositions));
}



//
// std::vector<BoxCoordWithPadding> splitImageHomogeneous(
//     const CuboidShape& workShape,
//     const Padding& cubePadding,
//     const CuboidShape& imageOriginalShape,
//     const PaddingType& imagePadding)
//     {
//     // workshape is the subimage + padding, this depends mainly on performance
//     CuboidShape subimageShape = workShape - cubePadding.before - cubePadding.after;
//     std::vector<BoxCoordWithPadding> cubePositions;
//
//     // true divide and also the padding doesnt apply to the first cube (which gets no padding before) and the last cube which gets no padding after)
//     // so thats why the cubePading is removed one time in each dimension on
//     int cubesInDepth = std::max(1,(imageOriginalShape.depth - cubePadding.getPaddingDepthTotal() + subimageShape.depth - 1) / subimageShape.depth);
//     int cubesInWidth = std::max(1,(imageOriginalShape.width - cubePadding.getPaddingWidthTotal() + subimageShape.width - 1) / subimageShape.width);
//     int cubesInHeight = std::max(1,(imageOriginalShape.height - cubePadding.getPaddingHeightTotal() + subimageShape.height - 1) / subimageShape.height);
//
//     // Calculate total number of cubes
//     int totalCubes = cubesInDepth * cubesInWidth * cubesInHeight;
//     cubePositions.reserve(totalCubes);
//     CuboidShape currentPos(0,0,0);
//
//     for (int d = 0; d < cubesInDepth; ++d) {
//         for (int h = 0; h < cubesInHeight; ++h) {
//             for (int w = 0; w < cubesInWidth; ++w) {
//
//                 // Calculate remaining size for this cube
//                 CuboidShape remainingSize(
//                     std::min(subimageShape.width, imageOriginalShape.width - currentPos.width),
//                     std::min(subimageShape.height, imageOriginalShape.height - currentPos.height),
//                     std::min(subimageShape.depth, imageOriginalShape.depth - currentPos.depth)
//                 );
//
//                 // Skip if no remaining size (shouldn't happen with proper calculation)
//                 if (remainingSize.depth <= 0 || remainingSize.width <= 0 || remainingSize.height <= 0) {
//                     continue;
//                 }
//
//                 // Determine actual cube positions - use overlap for boundary cubes
//                 CuboidShape actualPos = currentPos;
//                 CuboidShape actualDimensions = subimageShape;
//                 Padding adjustedPadding = cubePadding;
//
//
//                 BoxCoord cube;
//                 cube.position = actualPos;
//                 cube.dimensions = actualDimensions;
//
//                 BoxCoordWithPadding cubeWithPadding;
//                 cubeWithPadding.box = cube;
//                 cubeWithPadding.padding = adjustedPadding;
//
//                 cubePositions.push_back(std::move(cubeWithPadding));
//
//
//                 // Calculate current position in original image coordinates
//                 currentPos = currentPos + actualDimensions; //TODO this is wrong!
//             }
//         }
//     }
//
//     return cubePositions;
// }
//
