#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
#include "backend/ComplexData.h"
#include "deconvolution/ImageMap.h"
#include "deconvolution/deconvolutionStrategies/ComputationalPlan.h"
class PaddedImage;
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

    void removePadding(std::vector<cv::Mat>& image, const Padding& padding);
    void cropToOriginalSize(std::vector<cv::Mat>& image, const RectangleShape& originalSize);

}