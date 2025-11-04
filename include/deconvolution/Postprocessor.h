#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
#include "backend/ComplexData.h"
#include "deconvolution/ImageMap.h"

namespace Postprocessor{
    std::vector<cv::Mat> mergeImage(
        const std::vector<std::vector<cv::Mat>>& cubes,
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape,
        const RectangleShape& imageShapePadded,
        const RectangleShape& cubeShapePadded
    );


    void insertCubeInImage(
        std::vector<cv::Mat>& cube,
        std::vector<cv::Mat>& image,
        BoxCoord destBox,
        RectangleShape padding
    );
    
    void removePadding(std::vector<cv::Mat>& image, const RectangleShape& padding);
    void cropToOriginalSize(std::vector<cv::Mat>& image, const RectangleShape& originalSize);

}