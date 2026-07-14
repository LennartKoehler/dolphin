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
#include <vector>
#include <queue>

#include <tiffio.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffExceptions.h"
#include "dolphin_image/ImageMetaData.h"


class TiffWriter : public ImageWriter {
public:
    explicit TiffWriter(const std::string& filename, const CuboidShape& imageShape);

    void configure(WriterCompressionConfig compressionConfig) override;
    ~TiffWriter();

    bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const override;

    static bool writeToFile(const std::string& filename, const Image3D& image, WriterCompressionConfig config = {});


private:
    mutable std::mutex writerMutex;
    mutable CustomList<ImageBuffer> tileBuffer;
    std::string outputFilename;
    TIFF* tif;
    CuboidShape imageShape;
    WriterCompressionConfig compressionConfig;
    mutable size_t writtenToDepth = 0;
    mutable std::queue<int> readyToWriteQueue;

    bool writeToFile_(const std::string& filename, size_t z, size_t depth, const Image3D& layers) const;
    void createNewTile(const BoxCoordWithPadding& coord) const;
    bool isTileFull(const ImageBuffer& strip) const;
    bool writeTile(const std::string& filename, const ImageBuffer& strip) const;
    void copyToTile(const Image3D& image, const BoxCoordWithPadding& coord, int index) const;
    int getStripIndex(const BoxCoordWithPadding& coord) const;
    void processReadyToWriteQueue() const;

    static TIFF* openTiff(const char* filename, const CuboidShape& size);

    static ImageMetaData extractMetaData(const Image3D& image);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static void writeSliceToTiff(TIFF* tif, const Image3D& image, size_t sliceIndex, const ImageMetaData& metaData, const WriterCompressionConfig& compression);
    static void setTiffFields(TIFF* tif, const ImageMetaData& metaData, const WriterCompressionConfig& compression);

    static int getTargetItkType(const ImageMetaData& metadata);
    static void extractSliceData(const Image3D& image, size_t sliceIndex, std::vector<float>& sliceData);
    static void convertSliceDataToTargetType(const std::vector<float>& sourceData,
                                           std::vector<uint8_t>& targetData,
                                           size_t width, size_t height,
                                           const ImageMetaData& metadata);

};
