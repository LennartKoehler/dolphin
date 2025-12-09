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

#include "HyperstackImage.h"
#include <iostream>

void Hyperstack::printMetadata() const {
    std::cout << "[METADATA]" << std::endl;
    std::cout << "Image Type: " << metaData.imageType << std::endl;
    std::cout << "Name: " << metaData.filename << std::endl;
    std::cout << "Image Width: " << metaData.imageWidth << std::endl;
    std::cout << "Image Length: " << metaData.imageLength << std::endl;
    std::cout << "Frame Count: " << metaData.frameCount << std::endl;
    std::cout << "Resolution Unit: " << metaData.resolutionUnit << std::endl;
    std::cout << "Samples Per Pixel: " << metaData.samplesPerPixel << std::endl;
    std::cout << "Bits Per Sample: " << metaData.bitsPerSample << std::endl;
    std::cout << "Photometric Interpretation: " << metaData.photometricInterpretation << std::endl;
    std::cout << "Linearized Channels: " << metaData.linChannels << std::endl;
    std::cout << "Planar Configuration: " << metaData.planarConfig << std::endl;
    std::cout << "Total Images: " << metaData.totalImages << std::endl;
    std::cout << "Slices: " << metaData.slices << std::endl;
    std::cout << "Data Type: " << metaData.dataType << std::endl;
    std::cout << "X Resolution: " << metaData.xResolution << std::endl;
    std::cout << "Y Resolution: " << metaData.yResolution << std::endl;
    std::cout << "Description: " << metaData.description << std::endl;

}
