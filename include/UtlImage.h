#pragma once

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <fftw3.h>


namespace UtlImage {
    // find global min/max value of image pixel values
    void findGlobalMinMax(const std::vector<cv::Mat>& images, double& globalMin, double& globalMax);
    // normalize an image that all values sum equal 1
    void normalizeToSumOne(std::vector<cv::Mat>& psf);
    // checks for valid float(32) value (overlfloat protection)
    bool isValidForFloat(fftw_complex* fftwData, size_t size);

    }