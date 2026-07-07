
#pragma once

#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"



class TiffReaderWriterPair : public ImageReaderWriterPair{
public:
    TiffReaderWriterPair(const std::string& filenameInput, int channel, const std::string& filenameOutput);
    virtual std::future<PaddedImage> getSubimage(const BoxCoordWithPadding& box) const override;
    virtual const ImageMetaData& getMetaData() const override;
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const override;

    TiffReader reader;
    TiffWriter writer;
};

