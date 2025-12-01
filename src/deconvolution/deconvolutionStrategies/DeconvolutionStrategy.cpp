#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategy.h"




std::vector<BoxCoord> splitImageHomogeneous(
    const RectangleShape& subimageShape,
    const RectangleShape& imageOriginalShape)
{
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

