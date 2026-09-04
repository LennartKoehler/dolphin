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

#include "dolphin_image/HelperClasses.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphinbackend/CuboidShape.h"
#include <algorithm>
#include <vector>
#include <stdexcept>


void adjustCubeToBoundaries(
    BoxCoordWithPadding& cube,
    const CuboidShape& imageOriginalShape,
    const CuboidShape& remainingSize,
    const Padding& cubePadding,
    const Padding& imagePadding) {

    // Clamp if cube larger than image in any dimension
    if (cube.box.dimensions.width > imageOriginalShape.width)
        cube.box.dimensions.width = imageOriginalShape.width;
    if (cube.box.dimensions.height > imageOriginalShape.height)
        cube.box.dimensions.height = imageOriginalShape.height;
    if (cube.box.dimensions.depth > imageOriginalShape.depth)
        cube.box.dimensions.depth = imageOriginalShape.depth;

    // Shift-back for last cube in each dimension to prevent going out of bounds
    if (remainingSize.width < cube.box.dimensions.width && remainingSize.width > 0)
        cube.box.position.width -= static_cast<int64_t>(cube.box.dimensions.width - remainingSize.width);
    if (remainingSize.height < cube.box.dimensions.height && remainingSize.height > 0)
        cube.box.position.height -= static_cast<int64_t>(cube.box.dimensions.height - remainingSize.height);
    if (remainingSize.depth < cube.box.dimensions.depth && remainingSize.depth > 0)
        cube.box.position.depth -= static_cast<int64_t>(cube.box.dimensions.depth - remainingSize.depth);

    // Determine boundary status after shift-back
    bool atStartX = (cube.box.position.width == 0);
    bool atEndX = (cube.box.position.width + static_cast<int64_t>(cube.box.dimensions.width) >= static_cast<int64_t>(imageOriginalShape.width));
    bool atStartY = (cube.box.position.height == 0);
    bool atEndY = (cube.box.position.height + static_cast<int64_t>(cube.box.dimensions.height) >= static_cast<int64_t>(imageOriginalShape.height));
    bool atStartZ = (cube.box.position.depth == 0);
    bool atEndZ = (cube.box.position.depth + static_cast<int64_t>(cube.box.dimensions.depth) >= static_cast<int64_t>(imageOriginalShape.depth));

    // Set padding: imagePadding at image boundary, cubePadding for interior overlap
    cube.padding.before.width  = atStartX ? imagePadding.before.width  : cubePadding.before.width;
    cube.padding.after.width   = atEndX   ? imagePadding.after.width   : cubePadding.after.width;
    cube.padding.before.height = atStartY ? imagePadding.before.height : cubePadding.before.height;
    cube.padding.after.height  = atEndY   ? imagePadding.after.height  : cubePadding.after.height;
    cube.padding.before.depth  = atStartZ ? imagePadding.before.depth  : cubePadding.before.depth;
    cube.padding.after.depth   = atEndZ   ? imagePadding.after.depth   : cubePadding.after.depth;
}

// add new cube recursively
void addCubeRecursion(
    std::vector<BoxCoordWithPadding>& cubePositions,
    BoxCoordWithPadding& currentCube,
    const CuboidShape& imageOriginalShape,
    const Padding& cubePadding,
    const Padding& imagePadding) {

    assert(currentCube.box.dimensions.getVolume() > 0);

    // next row
    if (currentCube.box.position.width >= static_cast<int64_t>(imageOriginalShape.width)){
        currentCube.box.position.width = 0;
        currentCube.box.position.height += currentCube.box.dimensions.height;
        addCubeRecursion(cubePositions, currentCube, imageOriginalShape, cubePadding, imagePadding);
        return;
    }
    // next slice
    if (currentCube.box.position.height >= static_cast<int64_t>(imageOriginalShape.height)){
        currentCube.box.position.height = 0;
        currentCube.box.position.depth += currentCube.box.dimensions.depth;
        addCubeRecursion(cubePositions, currentCube, imageOriginalShape, cubePadding, imagePadding);
        return;
    }
    // were done
    if (currentCube.box.position.depth >= static_cast<int64_t>(imageOriginalShape.depth))
        return;

    CuboidShape remainingSize = imageOriginalShape - currentCube.box.position;

    BoxCoordWithPadding cubeToPush = currentCube;

    adjustCubeToBoundaries(cubeToPush, imageOriginalShape, remainingSize, cubePadding, imagePadding);

    cubePositions.push_back(cubeToPush);

    // next cube (column) — advance by nominal (unmutated) cube size
    currentCube.box.position.width += currentCube.box.dimensions.width;
    addCubeRecursion(cubePositions, currentCube, imageOriginalShape, cubePadding, imagePadding);
}


template <typename T>
std::array<size_t, 3> sort_indexes(const std::array<T*, 3> &v) {

    // initialize original index locations
    std::array<size_t, 3> idx;
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) {return *v[i1] > *v[i2];});

    return idx;
}
bool decreaseSize(std::array<size_t*, 3>& tempCubeAccessor, int dimension, const CuboidShape& minSize){

    size_t newSize = previousSmooth(*tempCubeAccessor[dimension]);
    if (newSize >= minSize.getArray()[dimension]){
        *(tempCubeAccessor[dimension]) = newSize;
        return true;
    }
    return false;
}

bool decreaseLargestDim(std::array<size_t*, 3>& tempCubeAccessor, const CuboidShape& minSize){

    std::array<size_t, 3> sortedIndices = sort_indexes<size_t>(tempCubeAccessor);
    for (const auto dimIndex : sortedIndices){
        size_t newSize = previousSmooth(*tempCubeAccessor[dimIndex]);
        if (newSize >= minSize.getArray()[dimIndex]){
            *(tempCubeAccessor[dimIndex]) = newSize;
            return true;
        }
    }
    return false;
}

std::vector<BoxCoordWithPadding> reduceSizeWhileKeepingNCubes(
        CuboidShape currentMaxSize,
        const CuboidShape& imageOriginalShape,
        const Padding& cubePadding,
        const Padding& imagePadding,
        const CuboidShape& minSize,
        size_t targetCubeCount,
        std::vector<BoxCoordWithPadding> cubePositions
    ){
    std::array<size_t*, 3> tempCubeAccessor  = currentMaxSize.getReference();

    assert(currentMaxSize >= minSize && "Input size already below minimum");

    for (int dim = 0; dim < 3; dim++) {
        while (true) {
            CuboidShape saved = currentMaxSize;

            if (!decreaseSize(tempCubeAccessor, dim, minSize))
                break;

            CuboidShape cubeSizeToUse = currentMaxSize - cubePadding.before - cubePadding.after;

            BoxCoordWithPadding startCube{
                BoxCoord{CuboidShape(0,0,0), cubeSizeToUse},
                cubePadding
            };

            std::vector<BoxCoordWithPadding> newCubes;
            addCubeRecursion(
                newCubes,
                startCube,
                imageOriginalShape,
                cubePadding,
                imagePadding);

            if (newCubes.size() > targetCubeCount) {
                currentMaxSize = saved;
                break;
            }

            cubePositions = std::move(newCubes);
        }
    }

    return cubePositions;
}

// since there are so many competing conditions for the cubes like maxSize (bc of memory), min number cubes(to e.g. use all devices)
// but also keep the size a smooth number for fftw, and somewhat "dynamic" padding if at an edge or not etc.
// so this is more of a just try out a bunch and when all conditions are sufficiently met then keep that plan
// i assume one could have also had a more complicated "model" of all the interactions and get a cube distribution that way
Result<std::vector<BoxCoordWithPadding>> splitImageHomogeneous(
    const Padding& cubePadding,
    const Padding& imagePadding,
    const CuboidShape& imageOriginalShape,
    const size_t& maxVolumePerCube,
    const size_t& minNumberCubes,
    const CuboidShape& minSize)
    {

    assert(minSize > cubePadding.getTotalPadding());

    CuboidShape currentMaxSize = imageOriginalShape + imagePadding.before + imagePadding.after;

    currentMaxSize.setMin(minSize); // because it has to be atleast as big as the psf

    // get next smooth size for faster fftw
    currentMaxSize.width = nextSmooth(currentMaxSize.width);
    currentMaxSize.height = nextSmooth(currentMaxSize.height);
    currentMaxSize.depth = nextSmooth(currentMaxSize.depth);

    std::array<size_t*, 3> tempCubeAccessor  = currentMaxSize.getReference();

    std::vector<BoxCoordWithPadding> cubePositions;

    while (true){

        cubePositions.clear();

        CuboidShape cubeSizeToUse = currentMaxSize - cubePadding.before - cubePadding.after;

        BoxCoordWithPadding startCube{
            BoxCoord{CuboidShape(0,0,0), cubeSizeToUse},
            cubePadding
        };

        if (startCube.getPaddedBox().dimensions.getVolume() < maxVolumePerCube){
            addCubeRecursion(
                cubePositions,
                startCube,
                imageOriginalShape,
                cubePadding,
                imagePadding);

            if (cubePositions.size() >= minNumberCubes)
                break;
        }

        bool success = decreaseLargestDim(tempCubeAccessor, minSize);
        if (!success)
        {
            return Result<std::vector<BoxCoordWithPadding>>::fail(
                "Not enough memory to fit the smallest possible cube: " + minSize.print());
        }
    }
    size_t targetCubeCount = cubePositions.size();
    CuboidShape inputPaddedShape = cubePositions[0].getPaddedShape();
    cubePositions = reduceSizeWhileKeepingNCubes(
        inputPaddedShape,
        imageOriginalShape,
        cubePadding,
        imagePadding,
        minSize,
        targetCubeCount,
        std::move(cubePositions)
    );

    return Result<std::vector<BoxCoordWithPadding>>::ok(std::move(cubePositions));
    }
