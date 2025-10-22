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

#include "deconvolution/Preprocessor.h"
#include <stdexcept>
#include <opencv2/core.hpp>

std::vector<std::vector<cv::Mat>> Preprocessor::splitImage(
    std::vector<cv::Mat>& image,
    const RectangleShape& subimageShape,
    const RectangleShape& imageOriginalShape,
    const RectangleShape& imageShapePadded,
    const RectangleShape& cubeShapePadded) {

    assert(imageOriginalShape >= subimageShape &&  "[ERROR] subimage has to be smaller than image");   

    // Calculate image padding from actual image dimensions vs imageShapePadded
    RectangleShape imagePadding = (imageShapePadded - imageOriginalShape) / 2;

    // Calculate actual cube padding amounts from cubeShapePadded
    int cubePaddingWidth = (cubeShapePadded.width - subimageShape.width) / 2;
    int cubePaddingHeight = (cubeShapePadded.height - subimageShape.height) / 2;
    int cubePaddingDepth = (cubeShapePadded.depth - subimageShape.depth) / 2;
    
    // Support asymmetric padding by calculating remaining padding
    int cubePaddingWidthEnd = cubeShapePadded.width - subimageShape.width - cubePaddingWidth;
    int cubePaddingHeightEnd = cubeShapePadded.height - subimageShape.height - cubePaddingHeight;
    int cubePaddingDepthEnd = cubeShapePadded.depth - subimageShape.depth - cubePaddingDepth;

    // Calculate number of cubes in each dimension
    int cubesInDepth = (imageOriginalShape.depth + subimageShape.depth - 1) / subimageShape.depth;
    int cubesInWidth = (imageOriginalShape.width + subimageShape.width - 1) / subimageShape.width;
    int cubesInHeight = (imageOriginalShape.height + subimageShape.height - 1) / subimageShape.height;
    
    // Calculate total number of cubes
    int totalCubes = cubesInDepth * cubesInWidth * cubesInHeight;

    // Ensure dimensions are valid
    if (imageOriginalShape.depth <= 0 || imageOriginalShape.height <= 0 || imageOriginalShape.width <= 0) {
        throw std::invalid_argument("Invalid image dimensions after accounting for padding.");
    }

    std::vector<std::vector<cv::Mat>> cubes;
    cubes.reserve(totalCubes);

    // Triple nested loop to iterate through all cube positions
    for (int d = 0; d < cubesInDepth; ++d) {
        for (int w = 0; w < cubesInWidth; ++w) {
            for (int h = 0; h < cubesInHeight; ++h) {
                
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

                // Process current cube - extract cubeShapePadded dimensions
                std::vector<cv::Mat> cube;
                cube.reserve(cubeShapePadded.depth);

                // Extract cube with asymmetric padding
                int depthStart = actualPos.depth - cubePaddingDepth;
                int depthEnd = actualPos.depth + subimageShape.depth + cubePaddingDepthEnd;

                for (int z = depthStart; z < depthEnd; ++z) {
                    
                    int widthStart = actualPos.width - cubePaddingWidth;
                    int heightStart = actualPos.height - cubePaddingHeight;
                    int widthEnd = actualPos.width + subimageShape.width + cubePaddingWidthEnd;
                    int heightEnd = actualPos.height + subimageShape.height + cubePaddingHeightEnd;

                    cv::Rect cubeSlice(widthStart, heightStart, widthEnd - widthStart, heightEnd - heightStart);
                    cv::Mat paddedSlice = image[z](cubeSlice).clone();

                    cube.push_back(paddedSlice);
                }

                cubes.push_back(std::move(cube));
            }
        }
    }
    
    return cubes;
}



void Preprocessor::padToShape(std::vector<cv::Mat>& image3D, const RectangleShape& targetShape, int borderType){
    if (image3D.empty()) return;
    
    int currentDepth = image3D.size();
    int currentHeight = image3D[0].rows;
    int currentWidth = image3D[0].cols;
    
    // Calculate total padding needed
    int totalDepthPadding = targetShape.depth - currentDepth;
    int totalHeightPadding = targetShape.height - currentHeight;
    int totalWidthPadding = targetShape.width - currentWidth;
    
    // If no padding needed, return early
    if (totalDepthPadding <= 0 && totalHeightPadding <= 0 && totalWidthPadding <= 0) {
        return;
    }
    
    // Handle depth padding (3D)
    if (totalDepthPadding > 0) {
        // Distribute padding: put extra padding at the end if odd
        int depthPaddingBefore = totalDepthPadding / 2;
        int depthPaddingAfter = totalDepthPadding - depthPaddingBefore;
        
        std::vector<cv::Mat> paddingBefore, paddingAfter;
        
        if (borderType == cv::BORDER_REFLECT) {
            // if the padding is larger than the image this would otherwise be negative
            int start = std::max(0, currentDepth - totalDepthPadding/2);
            // Before padding: reflect continuously if needed
            for (int i = start; i < depthPaddingBefore + start; ++i) {
                // Use modulo to continuously reflect through the image
                int sourceIndex = i % currentDepth;
                // For reflection, alternate between forward and backward
                if ((i / currentDepth) % 2 != 0) {
                    // Forward direction
                    paddingBefore.push_back(image3D[sourceIndex].clone());
                } else {
                    // Reverse direction  
                    paddingBefore.push_back(image3D[currentDepth - 1 - sourceIndex].clone());
                }
            }
            
            // After padding: reflect continuously if needed
            for (int i = 0; i < depthPaddingAfter; ++i) {
                int sourceIndex = i % currentDepth;
                if ((i / currentDepth) % 2 == 0) {
                    // Start from the end, going backward
                    paddingAfter.push_back(image3D[currentDepth - 1 - sourceIndex].clone());
                } else {
                    // Forward direction
                    paddingAfter.push_back(image3D[sourceIndex].clone());
                }
            }
        }
        else if (borderType == 0) {
            // Zero padding
            cv::Mat zeroMat = cv::Mat::zeros(currentHeight, currentWidth, image3D[0].type());
            paddingBefore.assign(depthPaddingBefore, zeroMat);
            paddingAfter.assign(depthPaddingAfter, zeroMat);
        }
        
        // Insert padding
        image3D.insert(image3D.begin(), paddingBefore.begin(), paddingBefore.end());
        image3D.insert(image3D.end(), paddingAfter.begin(), paddingAfter.end());
    }
    
    // Handle 2D padding (width/height) - OpenCV automatically handles continuous reflection
    if (totalHeightPadding > 0 || totalWidthPadding > 0) {
        int heightPaddingTop = totalHeightPadding / 2;
        int heightPaddingBottom = totalHeightPadding - heightPaddingTop;
        int widthPaddingLeft = totalWidthPadding / 2;
        int widthPaddingRight = totalWidthPadding - widthPaddingLeft;
        
        for (auto& layer : image3D) {
            cv::copyMakeBorder(layer, layer, 
                             heightPaddingTop, heightPaddingBottom,
                             widthPaddingLeft, widthPaddingRight, 
                             borderType);
        }
    }
}



void Preprocessor::expandToMinSize(std::vector<cv::Mat>& image, const RectangleShape& minSize) {
    if (image.empty()) return;
    
    int currentDepth = image.size();
    int currentHeight = image[0].rows;
    int currentWidth = image[0].cols;
    
    // Calculate padding needed for each dimension
    int depthPadding = std::max(0, minSize.depth - currentDepth);
    int heightPadding = std::max(0, minSize.height - currentHeight);
    int widthPadding = std::max(0, minSize.width - currentWidth);
    
    // Expand depth if needed
    if (depthPadding > 0) {
        image.reserve(minSize.depth);
        
        // Add slices at the end (could also mirror from beginning/end)
        for (int i = 0; i < depthPadding; ++i) {
            // Mirror from existing slices - use modulo to cycle through
            int sourceIndex = (currentDepth - 1) - (i % currentDepth);
            image.push_back(image[sourceIndex].clone());
        }
    }
    
    // Expand width and height if needed
    if (widthPadding > 0 || heightPadding > 0) {
        for (auto& layer : image) {
            cv::copyMakeBorder(layer, layer, 
                             0,              // top = 0 (no padding at top)
                             heightPadding,  // bottom = all height padding
                             0,              // left = 0 (no padding at left) 
                             widthPadding,   // right = all width padding
                             cv::BORDER_REFLECT_101);
        }
    }
}

