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

#include <string>
#include <cstdint>

class ImageMetaData {
public:
    // Image Attributes
    std::string filename;

    std::string imageType;
    std::string description;
    int imageWidth, imageLength = 0;
    int slices = 0;
    int frameCount = 0;
    uint16_t resolutionUnit = 0;
    uint16_t samplesPerPixel = 1; //num of channels
    uint16_t bitsPerSample = 0;//bit depth
    uint16_t photometricInterpretation = 0;
    int linChannels = 0;//in Description (linearized channels)
    uint16_t planarConfig = 0;
    int totalImages = -1;
     int dataType = 0; //calculated
    float xResolution, yResolution = 0.0f;


};

