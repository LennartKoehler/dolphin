#include <opencv2/core.hpp>
#include "UtlImage.h"

// find global min/max value of image pixel values
void UtlImage::findGlobalMinMax(const std::vector<cv::Mat>& images, double& globalMin, double& globalMax) {
    globalMin = std::numeric_limits<double>::max();
    globalMax = std::numeric_limits<double>::min();

    for (const auto& img : images) {
        double min, max;
        cv::minMaxLoc(img, &min, &max);
        if (min < globalMin) globalMin = min;
        if (max > globalMax) globalMax = max;
    }
}