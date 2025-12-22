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

#include "deconvolution/Postprocessor.h"
#include <stdexcept>
#include <functional>
#include "HelperClasses.h"
#include "itkImage.h"
#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageRegionIterator.h"

void Postprocessor::insertCubeInImage(
    const Image3D& cube,
    const BoxCoord& cubeBox,
    Image3D& image,
    const BoxCoord& srcBox
){

    for (int zCube = cubeBox.position.depth; zCube < cubeBox.position.depth + cubeBox.dimensions.depth; zCube++){
        cv::Rect roi(cubeBox.position.width, cubeBox.position.height, cubeBox.dimensions.width, cubeBox.dimensions.height);
        cv::Mat srcSlice = cube.slices[zCube](roi);
        // Define where it goes in the big image
        cv::Rect dstRoi(srcBox.position.width, srcBox.position.height, srcBox.dimensions.width, srcBox.dimensions.height);

        srcSlice.copyTo(image.slices[srcBox.position.depth + zCube - cubeBox.position.depth](dstRoi));
    }
}

// void Postprocessor::insertLabeledCubeInImage(
//     const PaddedImage& cube,
//     Image3D& outputImage, 
//     const BoxCoord& outputImageROI,
//     const BoxCoord& labeledImageROI,
//     const Label& labelgroup
// ){
//     for (int zCube = cube.padding.before.depth; zCube < outputImageROI.dimensions.depth + cube.padding.before.depth; zCube++){
//         cv::Rect roi(cube.padding.before.width, cube.padding.before.height, outputImageROI.dimensions.width, outputImageROI.dimensions.height);
//         cv::Mat srcSlice = cube.image.slices[zCube](roi);
        
//         // Define where it goes in the big image
//         cv::Rect dstRoi(outputImageROI.position.width, outputImageROI.position.height, outputImageROI.dimensions.width, outputImageROI.dimensions.height); 
//         int outputZ = outputImageROI.position.depth + zCube - cube.padding.before.depth;

//         cv::Rect labelRoi(labeledImageROI.position.width, labeledImageROI.position.height, labeledImageROI.dimensions.width, labeledImageROI.dimensions.height);
//         int labelZ = labeledImageROI.position.depth + zCube - cube.padding.before.depth;
        
//         cv::Mat mask = labelgroup.getMask(labelRoi, labelZ);
//         // Copy only where mask is true
//         srcSlice.copyTo(outputImage.slices[outputZ](dstRoi), mask);
//         // srcSlice.copyTo(outputImage[outputZ](dstRoi)); //TESTVALUE
//     }
    
    
// }

itk::Image<float, 3>::Pointer convertImage(const Image3D& image){
    using PixelType = float;
    constexpr unsigned int Dimension = 3;
    using ImageType = itk::Image<PixelType, Dimension>;

    ImageType::Pointer image3D = ImageType::New();
    RectangleShape imageShape = image.getShape();
    int width, height, depth;
    width = imageShape.width;
    height = imageShape.height;
    depth = imageShape.depth;

    ImageType::SizeType size;
    size[0] = width;
    size[1] = height;
    size[2] = depth;

    ImageType::IndexType start;
    start.Fill(0);

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    image3D->SetRegions(region);
    image3D->Allocate();
    image3D->FillBuffer(0);

    // Copy data slice by slice
    for (int z = 0; z < depth; ++z) {
        const cv::Mat& slice = image.slices[z];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ImageType::IndexType index = { x, y, z };
                float pixelValue = 0.0f;
                
                // Handle different OpenCV data types
                int cv_type = slice.type();
                if (cv_type == CV_32F || cv_type == CV_32FC1) {
                    pixelValue = slice.at<float>(y, x);
                } else if (cv_type == CV_8U || cv_type == CV_8UC1) {
                    pixelValue = static_cast<float>(slice.at<uint8_t>(y, x));
                } else if (cv_type == CV_16U || cv_type == CV_16UC1) {
                    pixelValue = static_cast<float>(slice.at<uint16_t>(y, x));
                } else {
                    // Default fallback - try to convert to float
                    cv::Mat convertedSlice;
                    slice.convertTo(convertedSlice, CV_32F);
                    pixelValue = convertedSlice.at<float>(y, x);
                }
                
                image3D->SetPixel(index, pixelValue);
            }
        }
    }
    return image3D;
}

Image3D convertItkImageToImage3D(const itk::Image<float, 3>::Pointer& itkImage) {
    using PixelType = float;
    constexpr unsigned int Dimension = 3;
    using ImageType = itk::Image<PixelType, Dimension>;

    // Get image dimensions
    ImageType::SizeType size = itkImage->GetLargestPossibleRegion().GetSize();
    int width = size[0];
    int height = size[1];
    int depth = size[2];

    // Create vector to hold OpenCV Mat slices
    std::vector<cv::Mat> slices;
    slices.reserve(depth);

    // Convert slice by slice
    for (int z = 0; z < depth; ++z) {
        // Create OpenCV Mat for this slice
        cv::Mat slice(height, width, CV_32F);
        
        // Copy pixel data from ITK to OpenCV
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ImageType::IndexType index = { x, y, z };
                float pixelValue = itkImage->GetPixel(index);

                slice.at<float>(y, x) = pixelValue;
            }
        }
        
        slices.push_back(slice);
    }

    return Image3D(std::move(slices));
}

// this also merges the images
Image3D Postprocessor::addFeathering(
    std::vector<ImageMaskPair>& pairs,
    int radius,
    double epsilon
){
    
    using PixelType = float;
    constexpr unsigned int Dimension = 3;

    using ImageType = itk::Image<PixelType, Dimension>;
    using DistanceFilterType = itk::DanielssonDistanceMapImageFilter<ImageType, ImageType>;

    using IteratorType = itk::ImageRegionIterator<ImageType>;

    struct ImageHelper{
        ImageType::Pointer image;
        ImageType::Pointer mask;
        IteratorType imageIt;
        IteratorType maskIt;
    };



    std::vector<ImageHelper> itkImageMasks;
    for (auto& pair : pairs){
        ImageType::Pointer image = convertImage(pair.image);
        ImageType::Pointer mask = convertImage(pair.mask);
        // Create iterators after storing the images, not before moving them
        ImageHelper helper;
        helper.image = image;
        helper.mask = mask;
        helper.imageIt = IteratorType(image, image->GetLargestPossibleRegion());
        // helper.maskIt = IteratorType(mask, mask->GetLargestPossibleRegion()); // fill in next loop
        itkImageMasks.push_back(std::move(helper));
    }


    ImageType::Pointer output = itkImageMasks[0].image;

    {
        for (auto& imagemask : itkImageMasks){
    

            // Create the distance filter
            DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
            distanceFilter->SetInput(imagemask.mask);
            // distanceFilter->SetUseImageSpacing(true); // takes voxel spacing into account if you have anisotropic voxels
            distanceFilter->Update();

            ImageType::Pointer distanceImage = distanceFilter->GetOutput();

            // Optionally, clip distances at 'radius' and convert back to 0-1 gradient
            itk::ImageRegionIterator<ImageType> it(distanceImage, distanceImage->GetLargestPossibleRegion());
            for (it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
                float dist = it.Get();
                if (dist > radius)
                    dist = radius;

                it.Set(1.0f - dist / radius); // gradient: 1 at mask, 0 at radius
            }
            imagemask.mask = distanceImage;
            // Recreate the mask iterator since we replaced the mask image
            imagemask.maskIt = IteratorType(imagemask.mask, imagemask.mask->GetLargestPossibleRegion());

        }
        

        IteratorType outIt(output,  output->GetLargestPossibleRegion());
        
        for(auto& image : itkImageMasks){
            image.imageIt.GoToBegin();
            image.maskIt.GoToBegin();
        }
        outIt.GoToBegin();

        std::vector<float> weights;
        std::vector<float> values;
        float sum = 1e-6f;  // Initialize sum
        float weight;

        while(!outIt.IsAtEnd()){
            weights.clear();  // Clear vectors at start of each iteration
            values.clear();
            sum = 1e-6f;      // Reset sum for each pixel
            
            for(auto& image : itkImageMasks){
                weight = image.maskIt.Get();
                values.push_back(image.imageIt.Get());
                weights.push_back(weight);
                sum += weight;

                ++image.imageIt;
                ++image.maskIt;
            }
            float resultValue = 0;
            if (sum > 0) {  // Avoid division by zero
                for(int i = 0; i < values.size(); i++){
                    resultValue += (weights[i] / sum) * values[i];  // Fixed formula for weighted blending
                }
            }
            outIt.Set(resultValue);
            ++outIt;

            
        }
    }
    return convertItkImageToImage3D(output);



}


void Postprocessor::removePadding(std::vector<cv::Mat>& image, const Padding& padding) {
    if (image.empty()) return;
    
    RectangleShape currentSize(image[0].cols, image[0].rows, image.size());
    


    RectangleShape cropAmount = padding.before + padding.after;

    // Crop depth if needed (remove from both ends like removePadding does)
    assert(image.size() > cropAmount.depth && "Image smaller than crop amount");
    
    // Remove from beginning
    if (padding.before.depth > 0) {
        image.erase(image.begin(), image.begin() + padding.before.depth);
    }
    
    // Remove from end
    if (padding.after.depth > 0) {
        image.erase(image.end() - padding.after.depth, image.end());
    }
   
    
    // Crop width and height if needed (remove from all sides like removePadding does)
    int newWidth = currentSize.width - cropAmount.width;
    int newHeight = currentSize.height - cropAmount.height;
    
    for (auto& slice : image) {
        // Crop symmetrically from all sides (like removePadding removes padding from all sides)
        cv::Rect cropRegion(padding.before.width, padding.before.height, newWidth, newHeight);
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
    
    Padding padding{cropStart, cropEnd};
    removePadding(image, padding);
}


// std::vector<cv::Mat> Postprocessor::mergeImage(
//     const std::vector<std::vector<cv::Mat>>& cubes,
//     const RectangleShape& subimageShape,
//     const RectangleShape& imageOriginalShape,
//     const RectangleShape& imageShapePadded,
//     const RectangleShape& cubeShapePadded
// ) {
//     // Calculate image padding
//     RectangleShape imagePadding = (imageShapePadded - imageOriginalShape) / 2;

//     // Calculate actual cube padding amounts from cubeShapePadded
//     int cubePaddingWidth = (cubeShapePadded.width - subimageShape.width) / 2;
//     int cubePaddingHeight = (cubeShapePadded.height - subimageShape.height) / 2;
//     int cubePaddingDepth = (cubeShapePadded.depth - subimageShape.depth) / 2;

//     // Calculate number of cubes in each dimension
//     int cubesInDepth = (imageOriginalShape.depth + subimageShape.depth - 1) / subimageShape.depth;
//     int cubesInWidth = (imageOriginalShape.width + subimageShape.width - 1) / subimageShape.width;
//     int cubesInHeight = (imageOriginalShape.height + subimageShape.height - 1) / subimageShape.height;

//     // Allocate padded image volume
//     std::vector<cv::Mat> merged(imageShapePadded.depth);
//     for (int z = 0; z < imageShapePadded.depth; ++z) {
//         merged[z] = cv::Mat::zeros(imageShapePadded.height, imageShapePadded.width, cubes[0][0].type());
//     }

//     int cubeIndex = 0;

//     // Triple nested loop to iterate through all cube positions (same order as splitImageHomogeneous)
//     for (int d = 0; d < cubesInDepth; ++d) {
//         for (int w = 0; w < cubesInWidth; ++w) {
//             for (int h = 0; h < cubesInHeight; ++h) {
                
//                 if (cubeIndex >= cubes.size()) {
//                     throw std::runtime_error("Not enough cubes supplied for merge!");
//                 }

//                 // Calculate current position in original image coordinates
//                 RectangleShape currentPos(
//                     imagePadding.width + w * subimageShape.width,
//                     imagePadding.height + h * subimageShape.height,
//                     imagePadding.depth + d * subimageShape.depth
//                 );

//                 // Calculate remaining size for this cube
//                 RectangleShape remainingSize(
//                     std::min(subimageShape.width, imageOriginalShape.width - w * subimageShape.width),
//                     std::min(subimageShape.height, imageOriginalShape.height - h * subimageShape.height),
//                     std::min(subimageShape.depth, imageOriginalShape.depth - d * subimageShape.depth)
//                 );

//                 // Skip if no remaining size (shouldn't happen with proper calculation)
//                 if (remainingSize.depth <= 0 || remainingSize.width <= 0 || remainingSize.height <= 0) {
//                     continue;
//                 }

//                 // Determine actual cube positions - use overlap for boundary cubes
//                 RectangleShape actualPos = currentPos;
                
//                 // If this would be the last cube and doesn't fit completely, shift it back to create overlap
//                 if (remainingSize.depth < subimageShape.depth && remainingSize.depth > 0) {
//                     actualPos.depth = currentPos.depth - (subimageShape.depth - remainingSize.depth);
//                 }
//                 if (remainingSize.width < subimageShape.width && remainingSize.width > 0) {
//                     actualPos.width = currentPos.width - (subimageShape.width - remainingSize.width);
//                 }
//                 if (remainingSize.height < subimageShape.height && remainingSize.height > 0) {
//                     actualPos.height = currentPos.height - (subimageShape.height - remainingSize.height);
//                 }

//                 const auto& cube = cubes[cubeIndex++];

//                 // Copy central cube region (ignore cubePadding)
//                 for (int z = cubePaddingDepth; z < subimageShape.depth + cubePaddingDepth; ++z) {
//                     int targetZ = actualPos.depth + (z - cubePaddingDepth);
//                     if (targetZ >= merged.size()) break;

//                     cv::Rect roi(cubePaddingWidth, cubePaddingHeight, subimageShape.width, subimageShape.height);
//                     cv::Mat srcSlice = cube[z](roi);

//                     // Define where it goes in the big image
//                     cv::Rect dstRoi(actualPos.width, actualPos.height, subimageShape.width, subimageShape.height);

//                     srcSlice.copyTo(merged[targetZ](dstRoi));
//                 }
//             }
//         }
//     }

//     return merged;
// }

// void Postprocessor::removePadding(std::vector<cv::Mat>& image, const RectangleShape& padding) {
//     // Remove the first and last layers according to padding
//     assert(image.size() > 2 * padding.depth && "Image only consists of padding, cant remove padding");
//     image.erase(image.begin(), image.begin() + padding.depth);
//     image.erase(image.end() - padding.depth, image.end());

//     // Crop the remaining layers in the X and Y dimensions
//     for (auto& slice : image) {
//         int newWidth = slice.cols - 2 * padding.width;
//         int newHeight = slice.rows - 2 * padding.height;
//         cv::Rect cropRegion(padding.width, padding.height, newWidth, newHeight);
//         slice = slice(cropRegion).clone();
//     }
// }

