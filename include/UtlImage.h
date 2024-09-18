#pragma once

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>

namespace UtlImage {
    // find global min/max value of image pixel values
    void findGlobalMinMax(const std::vector<cv::Mat>& images, double& globalMin, double& globalMax);
}