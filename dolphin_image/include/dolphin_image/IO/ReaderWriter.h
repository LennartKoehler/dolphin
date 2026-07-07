#pragma once
#include <optional>
#include <future>
#include <vector>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImageMetaData.h"

class ImageReader{
public:
    virtual ~ImageReader() = default;
    virtual std::future<PaddedImage> getSubimage(const BoxCoordWithPadding& box) const = 0;
    virtual void prefetch(const std::vector<BoxCoordWithPadding>& boxes) const {}
    virtual const ImageMetaData& getMetaData() const = 0;
    static std::string getFilename(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }
        return path.substr(pos + 1);
    }
};

class ImageWriter {
public:
    virtual ~ImageWriter() = default;
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const = 0;
};

struct ImageReaderWriterPair{

    virtual ~ImageReaderWriterPair() = default;
    virtual std::future<PaddedImage> getSubimage(const BoxCoordWithPadding& box) const = 0;
    virtual const ImageMetaData& getMetaData() const = 0;
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const = 0;
};
