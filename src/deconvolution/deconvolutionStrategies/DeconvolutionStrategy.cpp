#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategy.h"




std::vector<BoxCoordWithPadding> splitImageHomogeneous(
    const RectangleShape& subimageShape,
    const Padding& cubePadding,
    const RectangleShape& imageOriginalShape)
{
    std::vector<BoxCoordWithPadding> cubePositions;
    // Calculate number of cubes in each dimension
    int cubesInDepth = std::max(1,(imageOriginalShape.depth + subimageShape.depth - 1) / subimageShape.depth);
    int cubesInWidth = std::max(1,(imageOriginalShape.width + subimageShape.width - 1) / subimageShape.width);
    int cubesInHeight = std::max(1,(imageOriginalShape.height + subimageShape.height - 1) / subimageShape.height);
    
    // Calculate total number of cubes
    int totalCubes = cubesInDepth * cubesInWidth * cubesInHeight;
    cubePositions.reserve(totalCubes);

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
                RectangleShape actualDimensions = subimageShape;
                Padding adjustedPadding = cubePadding;
                
                // Check if padded cube exceeds image size and adjust accordingly
                RectangleShape paddedCubeSize = subimageShape + cubePadding.before + cubePadding.after;
                
                // If padded cube is larger than image in any dimension, adjust to make padding after larger while making dimensions of box smaller
                if (subimageShape.width > imageOriginalShape.width) {
                    actualDimensions.width = imageOriginalShape.width;
                    adjustedPadding.after.width = cubePadding.after.width + subimageShape.width - imageOriginalShape.width;
                }
                if (subimageShape.height > imageOriginalShape.height) {
                    actualDimensions.height = imageOriginalShape.height;
                    adjustedPadding.after.height = cubePadding.after.height + subimageShape.height - imageOriginalShape.height;
                }
                if (subimageShape.depth > imageOriginalShape.depth) {
                    actualDimensions.depth = imageOriginalShape.depth;
                    adjustedPadding.after.depth = cubePadding.after.depth + subimageShape.depth - imageOriginalShape.depth;
                }
                
                // If this would be the last cube and doesn't fit completely, shift it back to create overlap
                if (remainingSize.depth < actualDimensions.depth && remainingSize.depth > 0) {
                    actualPos.depth = currentPos.depth - (actualDimensions.depth - remainingSize.depth);
                }
                if (remainingSize.width < actualDimensions.width && remainingSize.width > 0) {
                    actualPos.width = currentPos.width - (actualDimensions.width - remainingSize.width);
                }
                if (remainingSize.height < actualDimensions.height && remainingSize.height > 0) {
                    actualPos.height = currentPos.height - (actualDimensions.height - remainingSize.height);
                }
                
                BoxCoord cube;
                cube.x = actualPos.width;
                cube.y = actualPos.height;
                cube.z = actualPos.depth;
                cube.dimensions = actualDimensions;
                
                BoxCoordWithPadding cubeWithPadding;
                cubeWithPadding.box = cube;
                cubeWithPadding.padding = adjustedPadding;
                
                cubePositions.push_back(std::move(cubeWithPadding));
            }
        }
    }

    return cubePositions;
}

