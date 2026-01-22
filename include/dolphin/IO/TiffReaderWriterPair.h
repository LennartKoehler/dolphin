
#pragma once

#include "ReaderWriter.h"
#include "TiffReader.h"
#include "TiffWriter.h"



class TiffReaderWriterPair : public ImageReaderWriterPair{

    TiffReaderWriterPair(const std::string& filenameInput, int channel, const std::string& filenameOutput);
    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const override;
    virtual const ImageMetaData& getMetaData() const override;
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const override;

    TiffReader reader;
    TiffWriter writer;
};

