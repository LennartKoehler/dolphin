#pragma once
#include "dolphin/Image3D.h"
#include "dolphin/ImageMetaData.h"

class ImageReader{
public:
    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const = 0;
    virtual const ImageMetaData& getMetaData() const = 0;
    static std::string getFilename(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path; // No directory separator found, return whole string
        }
        return path.substr(pos + 1);
    }
};

class ImageWriter {
public:
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const = 0;
};

class ImageReaderWriterPair{

    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const = 0;
    virtual const ImageMetaData& getMetaData() const = 0;
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const = 0;
};
