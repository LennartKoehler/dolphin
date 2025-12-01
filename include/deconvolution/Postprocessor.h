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
        PaddedImage& cube,
        std::vector<cv::Mat>& image,
        BoxCoord srcBox
    );
    void insertLabeledCubeInImage(
        const PaddedImage& cube,
        std::vector<cv::Mat>& image,
        const BoxCoord& srcBox,
        const LabelGroup& labelGroup
    );

    void removePadding(std::vector<cv::Mat>& image, const Padding& padding);
    void cropToOriginalSize(std::vector<cv::Mat>& image, const RectangleShape& originalSize);

}