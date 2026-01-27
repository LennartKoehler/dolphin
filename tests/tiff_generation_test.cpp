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

#include <iostream>
#include <string>
#include <vector>

#include <tiffio.h>
#include <filesystem>

#include "Image3D.h"
#include "ImageMetaData.h"
#include "RectangleShape.h"

/**
 * Creates a 3D TIFF image with a constant value throughout
 * 
 * @param filePath Path where to save the TIFF file
 * @param size 3D dimensions {width, height, depth}
 * @param value The constant value to fill the image with
 * @return true if successful, false otherwise
 */
bool createConstantValueTiff(const std::string& filePath, const int size[3], float value) {
    try {
        // Create directory if it doesn't exist
        std::filesystem::path dirPath = std::filesystem::path(filePath).parent_path();
        if (!dirPath.empty()) {
            std::filesystem::create_directories(dirPath);
        }

        // Open TIFF file for writing
        TIFF* tif = TIFFOpen(filePath.c_str(), "w");
        if (!tif) {
            std::cerr << "[ERROR] Cannot open TIFF file for writing: " << filePath << std::endl;
            return false;
        }

        // Set TIFF tags for the first image
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, size[0]);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, size[1]);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        
        // Set description metadata
        std::string description = "Generated test image\nConstant value: " + std::to_string(value) + 
                                "\nDimensions: " + std::to_string(size[0]) + "x" + std::to_string(size[1]) + "x" + std::to_string(size[2]);
        TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, description.c_str());

        // Create constant value image data
        std::vector<float> imageData(size[0] * size[1], value);

        // Write each slice (z-dimension)
        for (int z = 0; z < size[2]; ++z) {
            // Write scanlines for this slice
            for (int row = 0; row < size[1]; ++row) {
                if (TIFFWriteScanline(tif, imageData.data() + row * size[0], row, 0) < 0) {
                    std::cerr << "[ERROR] Failed to write scanline at slice " << z << ", row " << row << std::endl;
                    TIFFClose(tif);
                    return false;
                }
            }
            
            // Start new directory for next slice (multi-page TIFF)
            if (z < size[2] - 1) {
                TIFFWriteDirectory(tif);
                
                // Set TIFF tags for subsequent images
                TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, size[0]);
                TIFFSetField(tif, TIFFTAG_IMAGELENGTH, size[1]);
                TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
                TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
                TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
                TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
            }
        }

        TIFFClose(tif);
        std::cout << "[SUCCESS] Created constant value TIFF: " << filePath << std::endl;
        std::cout << "  Dimensions: " << size[0] << "x" << size[1] << "x" << size[2] << std::endl;
        std::cout << "  Constant value: " << value << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in createConstantValueTiff: " << e.what() << std::endl;
        return false;
    }
}



/**
 * Test function that creates multiple TIFF images with different parameters
 */
void runTiffGenerationTests() {
    std::cout << "\n=== TIFF Generation Tests ===" << std::endl;
    
    // Test parameters
    std::string outputDir = "../tests/labeledImage/test_tiff_output/";
    int testCases[][4] = {
        {32, 32, 20, 0},    // Small 3D volume
        {32, 32, 20, 1},    // Z-stack with zero
        // {128, 128, 5, 0},   // Medium 3D volume
        // {256, 256, 3, 255}  // Large 2D slices with max value
    };
    
    int numTestCases = sizeof(testCases) / sizeof(testCases[0]);
    
    // Test with direct TIFF writing
    std::cout << "\n--- Testing Direct TIFF Writing ---" << std::endl;
    for (int i = 0; i < numTestCases; ++i) {
        std::string filePath = outputDir + "constant_direct_" + std::to_string(i) + ".tif";
        int size[3] = {testCases[i][0], testCases[i][1], testCases[i][2]};
        float value = testCases[i][3];
        
        bool success = createConstantValueTiff(filePath, size, value);
        if (!success) {
            std::cerr << "[FAILED] Test case " << i << " failed" << std::endl;
        }
    }

    
    std::cout << "\n=== TIFF Generation Tests Completed ===" << std::endl;
}

// Main function for standalone testing
int main() {
    std::cout << "=== DOLPHIN TIFF Generation Test ===" << std::endl;
    
    // Run the tests
    runTiffGenerationTests();
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}