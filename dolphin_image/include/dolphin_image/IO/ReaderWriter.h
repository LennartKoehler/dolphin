#pragma once
#include <optional>
#include <future>
#include <vector>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImageMetaData.h"
#include "dolphin_image/ImagePadding.h"
#include "dolphin_image/Types/PaddingFillType.h"


class ImageReader{
public:
    virtual ~ImageReader() = default;
    virtual std::future<Image3D> getSubimage(const BoxCoord& box) const = 0;
    virtual void prefetch(const std::vector<BoxCoord>& boxes) const {}
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

// struct ImageReaderWriterPair{
//
//     virtual ~ImageReaderWriterPair() = default;
//     virtual std::future<PaddedImage> getSubimage(const BoxCoordWithPadding& box) const = 0;
//     virtual const ImageMetaData& getMetaData() const = 0;
//     virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const = 0;
// };


class ReaderHandler{
public:
    explicit ReaderHandler(std::unique_ptr<ImageReader> r, PaddingFillType paddingFillStrategy) : reader(std::move(r)), paddingFillStrategy(paddingFillStrategy){}
    const ImageMetaData& getMetaData() const { return reader->getMetaData(); }

    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const{
        BoxCoordWithPadding translatedRegion = translateRegion(box, reader->getMetaData().getShape());
        Image3D image = reader->getSubimage(translatedRegion.box).get();
        ImagePadding::padImage(image, translatedRegion.padding, paddingFillStrategy);
        PaddedImage result = PaddedImage{image, box.padding};
        return result;
    }

protected:
    BoxCoordWithPadding translateRegion(const BoxCoordWithPadding& requestedRegion, const CuboidShape& imageSize) const {
        BoxCoord paddedBox = requestedRegion.getPaddedBox();
        BoxCoord image = BoxCoord{CuboidShape{0,0,0}, imageSize};
        Padding padding = paddedBox.cropTo(image);
        return BoxCoordWithPadding{paddedBox, padding};
    }
    std::unique_ptr<ImageReader> reader;
    PaddingFillType paddingFillStrategy;
};
