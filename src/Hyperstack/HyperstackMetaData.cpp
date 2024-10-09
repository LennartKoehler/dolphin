#include "HyperstackImage.h"
#include <iostream>

void Hyperstack::printMetadata() const {
    std::cout << "[METADATA]" << std::endl;
    std::cout << "Image Type: " << metaData.imageType << std::endl;
    std::cout << "Name: " << metaData.name << std::endl;
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
