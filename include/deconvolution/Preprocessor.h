#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
#include "RectangleShape.h"

namespace Preprocessor{
    std::vector<std::vector<cv::Mat>> splitImage(
        std::vector<cv::Mat>& image,
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape,
        const RectangleShape& imageShapePadded,
        const RectangleShape& cubeShapePadded);

    void expandToMinSize(std::vector<cv::Mat>& image, const RectangleShape& minSize);


    void padToShape(std::vector<cv::Mat>& image3D, const RectangleShape& targetShape, int borderType);


}