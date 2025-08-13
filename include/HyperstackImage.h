#pragma once

#include <vector>
#include "Image3D.h"
#include "ImageMetaData.h"
#include "Channel.h"
#include "PSF.h"

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

    bool saveAsTifFile(const std::string &directoryPath);
    bool saveAsTifDir(const std::string& directoryPath);

    bool showSlice(int channel, int z);
    bool showChannel(int channel);

    void printMetadata() const;

    float getPixel(int channel, int x, int y, int z);
    bool isValid();

    Hyperstack convolve(PSF psf);

};

