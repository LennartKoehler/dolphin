#pragma once

#include "complexType.h"
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>



namespace UtlImage {
    // find global min/max value of image pixel values
    void findGlobalMinMax(const std::vector<cv::Mat>& images, double& globalMin, double& globalMax);
    // normalize an image that all values sum equal 1
    void normalizeToSumOne(std::vector<cv::Mat>& psf);
    // checks for valid float(32) value (overlfloat protection)
    bool isValidForFloat(complex* fftwData, size_t size);

    }