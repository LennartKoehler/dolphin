
#pragma once

#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"


class TiffReaderWriterPair {
public:
    TiffReaderWriterPair(const std::string& filenameInput, int channel, const std::string& filenameOutput);
    PaddedImage getSubimage(const BoxCoordWithPadding& box) const;
    const ImageMetaData& getMetaData() const;
    bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const;

    ReaderHandler reader;
    TiffWriter writer;
};

