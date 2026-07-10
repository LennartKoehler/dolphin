#include "dolphin_image/IO/TiffReaderWriterPair.h"


TiffReaderWriterPair::TiffReaderWriterPair(const std::string& filenameInput, int channel, const std::string& filenameOutput)
    : reader(std::make_unique<TiffReader>(filenameInput, channel)),
    writer(filenameOutput, reader.getMetaData().getShape()){}


PaddedImage TiffReaderWriterPair::getSubimage(const BoxCoordWithPadding& box) const {
    return reader.getSubimage(box);
}
const ImageMetaData& TiffReaderWriterPair::getMetaData() const{
    return reader.getMetaData();
}



bool TiffReaderWriterPair::setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const{
    return writer.setSubimage(image, coord);
}
