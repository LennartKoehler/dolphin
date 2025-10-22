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

#include <opencv2/core.hpp>
#include "UtlImage.h"

#include <iostream>

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

void UtlImage::normalizeToSumOne(std::vector<cv::Mat>& psf) {
    // Berechne die Summe aller Werte in der PSF
    double totalSum = 0.0;
    for (auto& slice : psf) {
        if (slice.type() != CV_32F) {
            std::cerr << "Error: Unsupported matrix type! " << slice.type() << std::endl;
        }
        totalSum += cv::sum(slice)[0]; // Summe über alle Pixel in jedem Slice
    }

    // Normiere jeden Slice, indem du ihn durch die Gesamtsumme teilst
    for (auto& slice : psf) {
        slice /= totalSum;
    }
}

bool UtlImage::isValidForFloat(complex* fftwData, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        // Überprüfen der Real- und Imaginärteile
        if (fftwData[i][0] < std::numeric_limits<float>::lowest() ||
            fftwData[i][0] > std::numeric_limits<float>::max() ||
            fftwData[i][1] < std::numeric_limits<float>::lowest() ||
            fftwData[i][1] > std::numeric_limits<float>::max()) {
            return false; // Ein Wert ist außerhalb des gültigen Bereichs
        }
    }
    return true; // Alle Werte sind gültig
}