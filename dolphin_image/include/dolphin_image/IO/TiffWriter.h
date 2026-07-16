/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knoll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include <tiffio.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffExceptions.h"
#include "dolphin_image/ImageMetaData.h"


class ITiffRegionWriter {
public:
    virtual ~ITiffRegionWriter() = default;
    virtual void writeSlice(TIFF* tif, const float* sliceData, const ImageMetaData& metaData) const = 0;
};

class TiffRegionWriterStripped : public ITiffRegionWriter {
public:
    void writeSlice(TIFF* tif, const float* sliceData, const ImageMetaData& metaData) const override;
};

class TiffRegionWriterTiled : public ITiffRegionWriter {
public:
    TiffRegionWriterTiled(uint32_t tileWidth, uint32_t tileLength);
    void writeSlice(TIFF* tif, const float* sliceData, const ImageMetaData& metaData) const override;
private:
    uint32_t tileWidth_;
    uint32_t tileLength_;
};


class TiffWriter : public ImageWriter {
public:
    explicit TiffWriter(const std::string& filename, const CuboidShape& imageShape);

    void configure(WriterCompressionConfig compressionConfig, WriterConfig writerConfig = {}) override;
    ~TiffWriter();

    bool setSubimage(const Image3D& image, const BoxCoord& coord,
                     const CuboidPosition& sourceOffset = CuboidPosition{0, 0, 0}) const override;

    static bool writeToFile(const std::string& filename, const Image3D& image,
                            WriterCompressionConfig compressionConfig = {},
                            WriterConfig writerConfig = {});


private:
    mutable std::mutex writerMutex;
    mutable CustomList<ImageBuffer> tileBuffer;
    std::string outputFilename;
    TIFF* tif;
    CuboidShape imageShape;
    WriterCompressionConfig compressionConfig;
    WriterConfig writerConfig_;
    mutable size_t writtenToDepth = 0;
    mutable std::map<int64_t, ImageBuffer> readyTiles_;

    std::unique_ptr<ITiffRegionWriter> regionWriter_;

    bool writeToFile_(size_t z, size_t depth, const Image3D& layers) const;
    void createNewTile(const BoxCoord& coord) const;
    bool isTileFull(const ImageBuffer& strip) const;
    void copyToTile(const Image3D& image, const BoxCoord& coord, const CuboidPosition& sourceOffset, int index) const;
    int getStripIndex(const BoxCoord& coord) const;
    void processReadyToWriteQueue() const;
    void writeTile(const ImageBuffer& tile) const;

    static TIFF* openTiff(const char* filename, const CuboidShape& size);

    static ImageMetaData extractMetaData(const Image3D& image);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static void setTiffFields(TIFF* tif, const ImageMetaData& metaData,
                              const WriterCompressionConfig& compression,
                              const WriterConfig& writerConfig);
    static int getTargetItkType(const ImageMetaData& metadata);
    static void convertSliceDataToTargetType(const std::vector<float>& sourceData,
                                           std::vector<uint8_t>& targetData,
                                           size_t width, size_t height,
                                           const ImageMetaData& metadata);
};
