#pragma once

#include <string>
#include <cstdint>

class ImageMetaData {
public:
    // Image Attributes
    std::string imageType;
    std::string name;
    std::string description;
    int imageWidth, imageLength = 0;
    int frameCount = 0;
    uint16_t resolutionUnit = 0;
    uint16_t samplesPerPixel = 1; //num of channels
    uint16_t bitsPerSample = 0;//bit depth
    uint16_t photometricInterpretation = 0;
    int linChannels = 0;//in Description (linearized channels)
    uint16_t planarConfig = 0;
    int totalImages = -1;
    int slices = 0;
    int dataType = 0; //calculated
    float xResolution, yResolution = 0.0f;


};

