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
    Image3D(std::vector<cv::Mat>&& slices){
        this->slices = std::move(slices);
    }

    Image3D(const Image3D& other){
        // Deep copy each slice
        slices.reserve(other.slices.size());
        for (const auto& slice : other.slices) {
            slices.push_back(slice.clone()); // Deep copy using clone()
        }
    }
    Image3D getSubimageCopy(const BoxCoord& coords){

        std::vector<cv::Mat> cube;
        cube.reserve(coords.dimensions.depth);
        
        for (int zCube = 0; zCube < coords.dimensions.depth; ++zCube) {
            

            // Define the ROI in the source image
            cv::Rect imageROI(coords.position.width, coords.position.height, coords.dimensions.width, coords.dimensions.height);
            int zImage = coords.position.depth;
            cv::Mat slice(coords.dimensions.height, coords.dimensions.width, CV_32F, cv::Scalar(0));

            // Copy from the source image ROI to the entire slice
            slices[zImage](imageROI).copyTo(slice);

            cube.push_back(std::move(slice));
        }
        return Image3D(std::move(cube));
    }
    std::vector<cv::Mat> slices;
    RectangleShape getShape() const ;
    float getPixel(int x, int y, int z);
    bool showSlice(int z);
    bool show();
};

