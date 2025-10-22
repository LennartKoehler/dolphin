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

#pragma once

#include "backend/ComplexData.h"
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