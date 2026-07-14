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
#include <optional>
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <typeindex>
#include <tiffio.h>
#include <itkImageRegionIterator.h>
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffExceptions.h"
#include "dolphin_image/IO/TiffHandlePool.h"
#include "dolphin_image/IO/ReaderThreadPool.h"
#include "dolphin_image/ImageMetaData.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphinbackend/IBackendMemoryManager.h"


struct BufferEntry {
    Image3D image;
    BoxCoord source;
};

using BufferIter = std::list<BufferEntry>::iterator;

struct PendingRead {
    BoxCoord source;
    std::vector<std::pair<BoxCoord, std::shared_ptr<std::promise<BufferIter>>>> waiters;
};



class ITiffRegionReader{
public:
    virtual BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const = 0 ;
    virtual void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel, size_t zImageOffset = 0) const = 0;
};

class TiffRegionReaderTiled : public ITiffRegionReader{
public:
    BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const override;
    template<typename T>
    static void convertTileRow(const char* tileRowData, std::vector<T>& rowData, size_t xOffset, size_t tilePixelWidth, const ImageMetaData& metaData, int channel);
    static void convertTileRowToFloat(const char* tileRowData, std::vector<float>& rowData, size_t xOffset, size_t tilePixelWidth, const ImageMetaData& metaData, int channel);
    virtual void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel, size_t zImageOffset = 0) const override;
};

class TiffRegionReaderStriped : public ITiffRegionReader{
public:
    BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const override;
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, size_t width, const ImageMetaData& metaData, int channel);
    virtual void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel, size_t zImageOffset = 0) const override;
};

class TiffRegionReaderScanline : public ITiffRegionReader{
public:
    BoxCoord computeReadSource(const ImageMetaData& metadata, const BoxCoord& box) const override;
    template<typename T>
    static void convertScanline(const char* scanlineData, std::vector<T>& rowData, size_t width, const ImageMetaData& metaData, int channel);
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, size_t width, const ImageMetaData& metaData, int channel);
    void readRegion(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int ifdchannel, int sppchannel, size_t zImageOffset = 0) const override;
};



template<typename T>
void TiffRegionReaderScanline::convertScanline(const char* scanlineData, std::vector<T>& rowData, size_t width, const ImageMetaData& metaData, int channel) {
    const uint16_t spp = metaData.samplesPerPixel;

    if (metaData.bitsPerSample == 8) {
        const uint8_t* data8 = reinterpret_cast<const uint8_t*>(scanlineData);
        for (size_t x = 0; x < width; x++) {
            rowData[x] = static_cast<T>(data8[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 16) {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(scanlineData);
        for (size_t x = 0; x < width; x++) {
            rowData[x] = static_cast<T>(data16[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 32) {
        const float* data32 = reinterpret_cast<const float*>(scanlineData);
        for (size_t x = 0; x < width; x++) {
            rowData[x] = static_cast<T>(data32[x * spp + channel]);
        }
    } else {
        throw TiffReadException("Unsupported bit depth: " + std::to_string(metaData.bitsPerSample) +
                                " (supported: 8, 16, 32)");
    }
}

template<typename T>
void TiffRegionReaderTiled::convertTileRow(const char* tileRowData, std::vector<T>& rowData, size_t xOffset, size_t tilePixelWidth, const ImageMetaData& metaData, int channel) {
    const uint16_t spp = metaData.samplesPerPixel;

    if (metaData.bitsPerSample == 8) {
        const uint8_t* data8 = reinterpret_cast<const uint8_t*>(tileRowData);
        for (size_t x = 0; x < tilePixelWidth; x++) {
            rowData[xOffset + x] = static_cast<T>(data8[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 16) {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(tileRowData);
        for (size_t x = 0; x < tilePixelWidth; x++) {
            rowData[xOffset + x] = static_cast<T>(data16[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 32) {
        const float* data32 = reinterpret_cast<const float*>(tileRowData);
        for (size_t x = 0; x < tilePixelWidth; x++) {
            rowData[xOffset + x] = static_cast<T>(data32[x * spp + channel]);
        }
    } else {
        throw TiffReadException("Unsupported bit depth: " + std::to_string(metaData.bitsPerSample) +
                                " (supported: 8, 16, 32)");
    }
}

class TiffReader : public ImageReader {
public:
    explicit TiffReader(const std::string& filename);
    ~TiffReader();

    void configure(int channel, ReaderConfig config = {}) override;
    static std::optional<Image3D> readTiffFile(const std::string& filename, int channel);
    static std::optional<ImageMetaData> readMetadata(const std::string& filename);

    Image3D getSubimage(const BoxCoord& box) const override;
    size_t getRequiredMemory(const CuboidShape& subimageSize) const override;
    // void prefetch(const std::vector<BoxCoord>& boxes) const override;
    const ImageMetaData& getMetaData() const override;

private:
    static size_t sizeOf(std::type_index type);

    mutable ImageMetaData metaData;
    int channel;
    std::string filename_;
    ReaderConfig config_;
    std::unique_ptr<ITiffRegionReader> regionReader;
    std::type_index dataType_{typeid(float)};

    std::unique_ptr<TiffHandlePool> handlePool_;
    std::unique_ptr<ReaderThreadPool> readerPool_;

    mutable std::unique_ptr<MemoryTracking> memoryTracker;

    mutable std::mutex mutex_;
    mutable std::condition_variable prefetchCv_;
    mutable std::list<BufferEntry> bufferedRegions_;
    mutable std::list<PendingRead> pendingReads_;
    mutable std::atomic<size_t> inFlightReads_{0};


    bool isMemoryAvailable(const CuboidShape& requestedSize) const;
    static std::unique_ptr<ITiffRegionReader> getRegionReader(const ImageMetaData& metadata);
    Image3D extractFromBuffer(const BoxCoord& coord, BufferEntry& entry) const;

    std::optional<BufferIter> tryGetFromBuffer(const BoxCoord& box, std::unique_lock<std::mutex>& lock) const;
    std::optional<BufferIter> tryWaitForInFlightRead(const BoxCoord& box, std::unique_lock<std::mutex>& lock) const;
    BufferIter readSubimage(const BoxCoord& box, std::unique_lock<std::mutex>& lock) const;

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
