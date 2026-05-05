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

/**
 * @file image_comparison_test.cpp
 * @brief Simple test for Image3D == Image3D comparison
 */

#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <limits>

#include <itkImage.h>
#include <itkPoint.h>
#include <itkIndex.h>

#include "dolphin/Image3D.h"
#include "dolphin/Logging.h"

// Create a 2x2x2 3D ITK image with custom values
Image3D create2x2x2Image(float fillValue = 0.0f) {
    // Define image size: 2x2x2
    constexpr unsigned int Dimension = 3;
    using PixelType = float;
    using ImageType = itk::Image<PixelType, Dimension>;

    // Set up the image region
    ImageType::IndexType start;
    start[0] = 0; start[1] = 0; start[2] = 0;

    ImageType::SizeType size;
    size[0] = 2; size[1] = 2; size[2] = 2;

    ImageType::RegionType region;
    region.SetIndex(start);
    region.SetSize(size);

    // Create the ITK image
    auto itkImage = ImageType::New();
    itkImage->SetRegions(region);
    itkImage->Allocate();
    itkImage->FillBuffer(fillValue);

    // Fill with custom values - iterate through all 8 voxels (2x2x2)
    // Z=0 plane: values 1, 2, 3, 4
    // Z=1 plane: values 5, 6, 7, 8
    ImageType::IndexType index;
    float customValues[2][2][2] = {
        {{1.0f, 2.0f}, {3.0f, 4.0f}},  // z=0
        {{5.0f, 6.0f}, {7.0f, 8.0f}}   // z=1
    };

    for (index[2] = 0; index[2] < 2; ++index[2]) {
        for (index[1] = 0; index[1] < 2; ++index[1]) {
            for (index[0] = 0; index[0] < 2; ++index[0]) {
                itkImage->SetPixel(index, customValues[index[2]][index[1]][index[0]]);
            }
        }
    }

    // Wrap ITK image in Image3D (using your wrapper class)
    return Image3D(itkImage);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;  // Suppress unused parameter warnings
    std::cout << "=== Image3D Comparison Test ===" << std::endl;

    try {
        Logging::init();

        // Step 1: Create first 2x2x2 image with custom values
        std::cout << "\n[Step 1] Creating first 2x2x2 image with custom values..." << std::endl;
        Image3D image1 = create2x2x2Image();
        std::cout << "Image1 shape: " << image1.getShape().width << " x "
                  << image1.getShape().height << " x " << image1.getShape().depth << std::endl;

        // Print values to verify
        std::cout << "Image1 values:" << std::endl;
        for (int z = 0; z < 2; ++z) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    std::cout << "  [" << x << "," << y << "," << z << "]: "
                              << image1.getPixel(x, y, z) << std::endl;
                }
            }
        }

        // Step 2: Create second identical image
        std::cout << "\n[Step 2] Creating second identical 2x2x2 image..." << std::endl;
        Image3D image2 = create2x2x2Image();
        image2.setPixel(0,0,0, 5.0f);
        std::cout << "Image2 created with same values." << std::endl;

        // Step 3: Compare the two images
        std::cout << "\n[Step 3] Comparing Image3D == Image3D..." << std::endl;

        if (image1 == image2) {
            std::cout << "\n=== Image Comparison Test PASSED ===" << std::endl;
            std::cout << "Both images are identical." << std::endl;
            return 0;
        } else {
            std::cout << "\n=== Image Comparison Test FAILED ===" << std::endl;
            std::cout << "Images are different!" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
