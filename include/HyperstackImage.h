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
#include "Image3D.h"
#include "ImageMetaData.h"
#include "Channel.h"
#include "psf/PSF.h"

class Hyperstack {
public:
    // Constructor and destructor defined
    Hyperstack();
    Hyperstack(const Hyperstack& rhs);
    ~Hyperstack();

    std::vector<Channel> channels;
    ImageMetaData metaData;

    bool readFromTifFile(const char *filename);
    bool readFromTifDir(const std::string& directoryPath);

    bool saveAsTifFile(const std::string &directoryPath) const ;
    bool saveAsTifDir(const std::string& directoryPath) const;

    bool showSlice(int channel, int z);
    bool showChannel(int channel) const;

    void printMetadata() const;

    float getPixel(int channel, int x, int y, int z);
    bool isValid();

    // Hyperstack convolve(PSF psf);

};

