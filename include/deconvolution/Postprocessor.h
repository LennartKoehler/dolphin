#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
#include "backend/ComplexData.h"
#include "deconvolution/ImageMap.h"
#include "deconvolution/deconvolutionStrategies/ComputationalPlan.h"
#include "itkImage.h"

class PaddedImage;

// Utility functions for ITK/Image3D conversion
itk::Image<float, 3>::Pointer convertImage(const Image3D& image);
Image3D convertItkImageToImage3D(const itk::Image<float, 3>::Pointer& itkImage);



namespace Postprocessor{
    std::vector<cv::Mat> mergeImage(
        const std::vector<std::vector<cv::Mat>>& cubes,
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape,
        const RectangleShape& imageShapePadded,
        const RectangleShape& cubeShapePadded
    );


    void insertCubeInImage(
        const Image3D& cube,
        const BoxCoord& cubeBox,
        Image3D& image,
        const BoxCoord& srcBox
    );
    
    void insertLabeledCubeInImage(
        const PaddedImage& cube,
        Image3D& image,
        const BoxCoord& srcBox,
        const BoxCoord& labeledImageROI,
        const Label& labelGroup
    );

    Image3D addFeathering(
        std::vector<ImageMaskPair>& pair,
        int radius,
        double epsilon
    );
    void removePadding(std::vector<cv::Mat>& image, const Padding& padding);
    void cropToOriginalSize(std::vector<cv::Mat>& image, const RectangleShape& originalSize);

}