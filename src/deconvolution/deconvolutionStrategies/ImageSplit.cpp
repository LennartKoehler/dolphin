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


BoxCoordWithPadding adjustCubeToBoundaries(
    const BoxCoord& cube,
    const CuboidShape& imageOriginalShape,
    const CuboidShape& remainingSize,
    const Padding& insidePadding,
    const Padding& outsidePadding) {

    BoxCoordWithPadding paddedCube;
    paddedCube.box = cube;

    for (size_t d = 0; d < 3; ++d) {
        // padding if cube at beginning
        // boundary faces get exactly imagePadding, interior faces at least cubePadding
        if (paddedCube.box.position.at(d) == 0){
            paddedCube.box.dimensions.at(d) += insidePadding.before.at(d) - outsidePadding.before.at(d);
            paddedCube.padding.before.at(d) = outsidePadding.before.at(d);
        }
        else paddedCube.padding.before.at(d) = insidePadding.before.at(d);

        // padding if cube at end
        // last cube per dimension: overflow goes into padding before (box stays on grid, boxes contiguous)
        // if the cube is larger than the entire image, then this will reduce the size (dimensions) of the cube and make that into padding
        //      this is because the dimension is used for the image primarily (e.g. writer and therefore has to match the image dimensions)
        if (remainingSize.at(d) <= paddedCube.box.dimensions.at(d) && remainingSize.at(d) > 0){
            paddedCube.box.dimensions.at(d) = remainingSize.at(d); // dimensions of cube doesnt go over edge
            paddedCube.padding.before.at(d) = insidePadding.before.at(d)
                + cube.dimensions.at(d) - remainingSize.at(d) // add the part that was removed from the dimension as padding before
                + insidePadding.after.at(d) - outsidePadding.after.at(d); // add how much the outsidePadding is larger than insidePadding

            paddedCube.padding.after.at(d) = outsidePadding.after.at(d); // set the padding after to the desired image padding

            if (outsidePadding.after.at(d) > paddedCube.padding.after.at(d)){
                // if image padding larger than paddedCube.padding then reduce the size by that amount to keep the total size (dimensions + padding) the same
                paddedCube.box.dimensions.at(d) -= outsidePadding.after.at(d) - outsidePadding.after.at(d);
            }
        }
        else paddedCube.padding.after.at(d) = insidePadding.after.at(d);
    }
    return paddedCube;
}

// add new cube recursively
void addCubeRecursion(
    std::vector<BoxCoordWithPadding>& cubePositions,
    BoxCoord& currentCube,
    BoxCoord& lastCube, //online used if the previous row or column where shifted due to edge conditions then need to shift the position start for the next
    const CuboidShape& imageOriginalShape,
    const Padding& cubePadding,
    const Padding& imagePadding) {

    assert(currentCube.dimensions.getVolume() > 0);

    // next row
    if (currentCube.position.width >= static_cast<int64_t>(imageOriginalShape.width)){
        currentCube.position.width = 0;
        currentCube.position.height += lastCube.dimensions.height;
        addCubeRecursion(cubePositions, currentCube, lastCube, imageOriginalShape, cubePadding, imagePadding);
        return;
    }
    // next slice
    if (currentCube.position.height >= static_cast<int64_t>(imageOriginalShape.height)){
        currentCube.position.height = 0;
        currentCube.position.depth += lastCube.dimensions.depth;
        addCubeRecursion(cubePositions, currentCube, lastCube, imageOriginalShape, cubePadding, imagePadding);
        return;
    }
    // were done
    if (currentCube.position.depth >= static_cast<int64_t>(imageOriginalShape.depth))
        return;

    CuboidShape remainingSize = imageOriginalShape - currentCube.position;


    BoxCoordWithPadding cubeToPush = adjustCubeToBoundaries(currentCube, imageOriginalShape, remainingSize, cubePadding, imagePadding);

    cubePositions.push_back(cubeToPush);

    // next cube (column) — advance by nominal (unmutated) cube size
    currentCube.position.width += cubeToPush.box.dimensions.width;
    addCubeRecursion(cubePositions, currentCube, cubeToPush.box, imageOriginalShape, cubePadding, imagePadding);
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

            BoxCoord startCube{
                BoxCoord{CuboidShape(0,0,0), cubeSizeToUse}
            };

            std::vector<BoxCoordWithPadding> newCubes;
            addCubeRecursion(
                newCubes,
                startCube,
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

        BoxCoord startCube{
            BoxCoord{CuboidShape(0,0,0), cubeSizeToUse}
        };

        if (startCube.dimensions.getVolume() + cubePadding.getTotalPadding().getVolume() < maxVolumePerCube){
            addCubeRecursion(
                cubePositions,
                startCube,
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
