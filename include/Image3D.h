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

#include <vector>
#include <opencv2/core/mat.hpp>
#include "HelperClasses.h"
class Image3D {
public:
    Image3D() = default;
    Image3D(const Image3D& other){
        // Deep copy each slice
        slices.reserve(other.slices.size());
        for (const auto& slice : other.slices) {
            slices.push_back(slice.clone()); // Deep copy using clone()
        }
    }
    std::vector<cv::Mat> slices;
    RectangleShape getShape() const ;
    float getPixel(int x, int y, int z);
    bool showSlice(int z);
    bool show();
};

