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
#include <numeric>
#include <vector>


void adjustCubeToBoundaries(
    BoxCoordWithPadding& cube,
    const CuboidShape& imageOriginalShape,
    const CuboidShape& remainingSize,
    const Padding& cubePadding,
    const Padding& imagePadding) {

    for (size_t d = 0; d < 3; ++d) {
        // boundary faces get exactly imagePadding, interior faces at least cubePadding
        if (cube.box.position.at(d) == 0){
            cube.box.dimensions.at(d) += cube.padding.before.at(d) - imagePadding.before.at(d);
            cube.padding.before.at(d) = imagePadding.before.at(d);
        }
        // if (cube.box.position.at(d) + static_cast<int64_t>(cube.box.dimensions.at(d)) == static_cast<int64_t>(imageOriginalShape.at(d))){
        //     cube.padding.before.at(d) += cube.padding.after.at(d) - imagePadding.after.at(d);
        //     cube.padding.after.at(d) = imagePadding.after.at(d);
        // }
        // last cube per dimension: overflow goes into padding before (box stays on grid, boxes contiguous)
        // if the cube is larger than the entire image, then this will reduce the size (dimensions) of the cube and make that into padding
        //      this is because the dimension is used for the image primarily (e.g. writer and therefore has to match the image dimensions)
        if (remainingSize.at(d) <= cube.box.dimensions.at(d) && remainingSize.at(d) > 0){
            size_t extra = cube.box.dimensions.at(d) - remainingSize.at(d)
                    - (cube.padding.after.at(d) - imagePadding.after.at(d)); // this is the same as the if condition above (image padding different than cube padding)
            cube.box.dimensions.at(d) -= extra;
            cube.padding.before.at(d) += extra;
            cube.padding.after.at(d) = imagePadding.after.at(d);
        }
    }
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
    currentCube.box.position.width += cubeToPush.box.dimensions.width;
    addCubeRecursion(cubePositions, currentCube, imageOriginalShape, cubePadding, imagePadding);
}


std::array<size_t, 3> sortDimensionsBySizeDesc(const CuboidShape& shape) {
    std::array<size_t, 3> idx;
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                [&shape](size_t i1, size_t i2) {return shape.at(i1) > shape.at(i2);});
    return idx;
}

bool decreaseSize(CuboidShape& size, int dimension, const CuboidShape& minSize){
    size_t newSize = previousSmooth(size.at(dimension));
    if (newSize >= minSize.at(dimension)){
        size.at(dimension) = newSize;
        return true;
    }
    return false;
}

bool decreaseLargestDim(CuboidShape& size, const CuboidShape& minSize){
    std::array<size_t, 3> sortedIndices = sortDimensionsBySizeDesc(size);
    for (const auto dimIndex : sortedIndices){
        size_t newSize = previousSmooth(size.at(dimIndex));
        if (newSize >= minSize.at(dimIndex)){
            size.at(dimIndex) = newSize;
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
    assert(currentMaxSize >= minSize && "Input size already below minimum");

    for (int dim = 0; dim < 3; dim++) {
        while (true) {
            CuboidShape saved = currentMaxSize;

            if (!decreaseSize(currentMaxSize, dim, minSize))
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
                currentMaxSize = saved; // max size is the previous max size as the most recent size reduction caused the number of cubes to go up (unwanted)
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
//
// imagePadding is the padding which the cubes should have at the edge of the image facing outwards, this might be less than cubePadding,
//      which is the padding cubes have on the inside of the image (so the overlap between neighboring cubes)
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
    currentMaxSize.transform([](size_t& d){ d = nextSmooth(d); });

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

        bool success = decreaseLargestDim(currentMaxSize, minSize);
        if (!success)
        {
            return Result<std::vector<BoxCoordWithPadding>>::fail(
                "Not enough memory to fit the smallest possible cube: " + minSize.print());
        }
    }
    size_t targetCubeCount = cubePositions.size();
    cubePositions = reduceSizeWhileKeepingNCubes(
        currentMaxSize,
        imageOriginalShape,
        cubePadding,
        imagePadding,
        minSize,
        targetCubeCount,
        std::move(cubePositions)
    );

    return Result<std::vector<BoxCoordWithPadding>>::ok(std::move(cubePositions));
}
