#include "deconvolution/Postprocessor.h"
#include <stdexcept>
#include <functional>

std::vector<cv::Mat> Postprocessor::mergeImage(
    const std::vector<std::vector<cv::Mat>>& cubes,
    const RectangleShape& subimageShape,
    const RectangleShape& imageOriginalShape,
    const RectangleShape& imageShapePadded,
    const RectangleShape& cubeShapePadded
) {
    // Calculate image padding
    RectangleShape imagePadding = (imageShapePadded - imageOriginalShape) / 2;

    // Calculate actual cube padding amounts from cubeShapePadded
    int cubePaddingWidth = (cubeShapePadded.width - subimageShape.width) / 2;
    int cubePaddingHeight = (cubeShapePadded.height - subimageShape.height) / 2;
    int cubePaddingDepth = (cubeShapePadded.depth - subimageShape.depth) / 2;

    // Calculate number of cubes in each dimension
    int cubesInDepth = (imageOriginalShape.depth + subimageShape.depth - 1) / subimageShape.depth;
    int cubesInWidth = (imageOriginalShape.width + subimageShape.width - 1) / subimageShape.width;
    int cubesInHeight = (imageOriginalShape.height + subimageShape.height - 1) / subimageShape.height;

    // Allocate padded image volume
    std::vector<cv::Mat> merged(imageShapePadded.depth);
    for (int z = 0; z < imageShapePadded.depth; ++z) {
        merged[z] = cv::Mat::zeros(imageShapePadded.height, imageShapePadded.width, cubes[0][0].type());
    }

    int cubeIndex = 0;

    // Triple nested loop to iterate through all cube positions (same order as splitImage)
    for (int d = 0; d < cubesInDepth; ++d) {
        for (int w = 0; w < cubesInWidth; ++w) {
            for (int h = 0; h < cubesInHeight; ++h) {
                
                if (cubeIndex >= cubes.size()) {
                    throw std::runtime_error("Not enough cubes supplied for merge!");
                }

                // Calculate current position in original image coordinates
                RectangleShape currentPos(
                    imagePadding.width + w * subimageShape.width,
                    imagePadding.height + h * subimageShape.height,
                    imagePadding.depth + d * subimageShape.depth
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

                const auto& cube = cubes[cubeIndex++];

                // Copy central cube region (ignore cubePadding)
                for (int z = cubePaddingDepth; z < subimageShape.depth + cubePaddingDepth; ++z) {
                    int targetZ = actualPos.depth + (z - cubePaddingDepth);
                    if (targetZ >= merged.size()) break;

                    cv::Rect roi(cubePaddingWidth, cubePaddingHeight, subimageShape.width, subimageShape.height);
                    cv::Mat srcSlice = cube[z](roi);

                    // Define where it goes in the big image
                    cv::Rect dstRoi(actualPos.width, actualPos.height, subimageShape.width, subimageShape.height);

                    srcSlice.copyTo(merged[targetZ](dstRoi));
                }
            }
        }
    }

    return merged;
}

void Postprocessor::removePadding(std::vector<cv::Mat>& image, const RectangleShape& padding) {
    // Remove the first and last layers according to padding
    assert(image.size() > 2 * padding.depth && "Image only consists of padding, cant remove padding");
    image.erase(image.begin(), image.begin() + padding.depth);
    image.erase(image.end() - padding.depth, image.end());

    // Crop the remaining layers in the X and Y dimensions
    for (auto& slice : image) {
        int newWidth = slice.cols - 2 * padding.width;
        int newHeight = slice.rows - 2 * padding.height;
        cv::Rect cropRegion(padding.width, padding.height, newWidth, newHeight);
        slice = slice(cropRegion).clone();
    }
}

void Postprocessor::cropToOriginalSize(std::vector<cv::Mat>& image, const RectangleShape& originalSize) {
    if (image.empty()) return;
    
    RectangleShape currentSize(image[0].cols, image[0].rows, image.size());
    
    // Calculate how much to crop from each dimension
    RectangleShape cropAmount(std::max(0, currentSize.width - originalSize.width),
                             std::max(0, currentSize.height - originalSize.height),
                             std::max(0, currentSize.depth - originalSize.depth));
    
    // For symmetric cropping, distribute evenly between start and end
    RectangleShape cropStart(cropAmount.width / 2, cropAmount.height / 2, cropAmount.depth / 2);
    RectangleShape cropEnd = cropAmount - cropStart;
    
    // Crop depth if needed (remove from both ends like removePadding does)
    if (cropAmount.depth > 0) {
        assert(image.size() > cropAmount.depth && "Image smaller than crop amount");
        
        // Remove from beginning
        if (cropStart.depth > 0) {
            image.erase(image.begin(), image.begin() + cropStart.depth);
        }
        
        // Remove from end
        if (cropEnd.depth > 0) {
            image.erase(image.end() - cropEnd.depth, image.end());
        }
    }
    
    // Crop width and height if needed (remove from all sides like removePadding does)
    if (cropAmount.width > 0 || cropAmount.height > 0) {
        int newWidth = currentSize.width - cropAmount.width;
        int newHeight = currentSize.height - cropAmount.height;
        
        for (auto& slice : image) {
            // Crop symmetrically from all sides (like removePadding removes padding from all sides)
            cv::Rect cropRegion(cropStart.width, cropStart.height, newWidth, newHeight);
            slice = slice(cropRegion).clone();
        }
    }
}