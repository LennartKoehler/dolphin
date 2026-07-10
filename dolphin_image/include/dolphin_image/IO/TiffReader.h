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
#include <future>
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <tiffio.h>
#include <itkImageRegionIterator.h>
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffExceptions.h"
#include "dolphin_image/IO/TiffHandlePool.h"
#include "dolphin_image/IO/ReaderThreadPool.h"
#include "dolphin_image/ImageMetaData.h"
#include "dolphin_image/Types/BoxCoord.h"


struct TiffReaderConfig {
    size_t numReaderThreads = 1;
    bool prefetchEnabled = false;
    size_t prefetchCount = 4;
};

struct BufferEntry {
    Image3D image;
    BoxCoord source;
};

struct PendingRead {
    BoxCoord source;
    std::vector<std::pair<BoxCoord, std::shared_ptr<std::promise<Image3D>>>> waiters;
};



class ITiffRegionReader{
public:
    virtual BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const = 0 ;
    virtual void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel) const = 0;
};

class TiffRegionReaderTiled : public ITiffRegionReader{
public:
    BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const override;
    static void convertTileRowToFloat(const char* tileRowData, std::vector<float>& rowData, size_t xOffset, size_t tilePixelWidth, const ImageMetaData& metaData, int channel);
    virtual void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel) const override;
};

class TiffRegionReaderStriped : public ITiffRegionReader{
public:
    BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const override;
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, size_t width, const ImageMetaData& metaData, int channel);
    virtual void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel) const override;
};

class TiffRegionReaderScanline : public ITiffRegionReader{
public:
    BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const override;
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, size_t width, const ImageMetaData& metaData, int channel);
    void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel) const override;
};



class TiffReader : public ImageReader {
public:
    explicit TiffReader(const std::string& filename, int channel, TiffReaderConfig config = {});
    ~TiffReader();

    static std::optional<Image3D> readTiffFile(const std::string& filename, int channel);
    static std::optional<ImageMetaData> readMetadata(const std::string& filename);

    std::future<Image3D> getSubimage(const BoxCoord& box) const override;
    void prefetch(const std::vector<BoxCoord>& boxes) const override;
    const ImageMetaData& getMetaData() const override;

private:
    mutable ImageMetaData metaData;
    int channel;
    std::string filename_;
    TiffReaderConfig config_;
    std::unique_ptr<ITiffRegionReader> regionReader;

    std::unique_ptr<TiffHandlePool> handlePool_;
    std::unique_ptr<ReaderThreadPool> readerPool_;

    mutable std::mutex mutex_;
    mutable std::condition_variable prefetchCv_;
    mutable std::list<BufferEntry> bufferedRegions_;
    mutable std::list<PendingRead> pendingReads_;
    mutable std::atomic<size_t> inFlightReads_{0};


    static std::unique_ptr<ITiffRegionReader> getRegionReader(const ImageMetaData& metadata);
    Image3D extractFromBuffer(const BoxCoord& coord, BufferEntry& entry) const;
    void executeRead(std::list<PendingRead>::iterator pendingIt, const BoxCoord& source) const;

    static void readSubimageFromTiffFile(TIFF* tiffile, const ITiffRegionReader* regionReader, const ImageMetaData& metaData, const BoxCoord& region, Image3D& layers, int channel);
    // static void readTiledSubimage(TIFF* tif, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel);
    // static void readStrippedSubimage(TIFF* tif, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel);
    // static void readScanlineSubimage(TIFF* tif, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel);
    static void readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, const BoxCoord& region, Image3D& layers, int channel);
    static ImageMetaData readMetadata_(const std::string& filename);

    static void resolveChannel(int channel, const ImageMetaData& metaData, int& ifdchannel, int& sppchannel);
    static void convertImageTo32F(Image3D& image, const ImageMetaData& metaData);
    static ImageMetaData extractMetadataFromTiff(TIFF*& tifFile);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);

    static int countTiffDirectories(TIFF* tif);
};
