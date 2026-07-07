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

#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffExceptions.h"
#include "dolphin_image/ImagePadding.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdarg>
#include <chrono>
#include <thread>
#include <itkImageRegionIterator.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace {

template<typename SliceFn>
void forEachZSlice(TIFF* tif, size_t z, size_t depth, const ImageMetaData& metaData, int ifdchannel, SliceFn&& sliceFn) {
    for (uint32_t zIndex = static_cast<uint32_t>(z); zIndex < static_cast<uint32_t>(z + depth); zIndex++) {
        int zIndexChannel = (zIndex * metaData.linChannels) + ifdchannel;
        if (!TIFFSetDirectory(tif, zIndexChannel)) {
            throw TiffReadException("Failed to set directory for z-slice " + std::to_string(zIndex) +
                                    " (channel " + std::to_string(zIndexChannel) + ")");
        }
        sliceFn(zIndex);
    }
}

}


TiffReader::TiffReader(const std::string& filename, int channel, TiffReaderConfig config)
    : channel(channel), filename_(filename), config_(config) {

    TIFFSetWarningHandler(customTifWarningHandler);

    try {
        metaData = readMetadata_(filename);
    } catch (const TiffException& e) {
        throw TiffFileOpenException(filename);
    }

    size_t numThreads = config_.numReaderThreads > 0 ? config_.numReaderThreads : 1;
    handlePool_ = std::make_unique<TiffHandlePool>(filename, numThreads);
    readerPool_ = std::make_unique<ReaderThreadPool>(numThreads);
}

TiffReader::~TiffReader() = default;


std::optional<Image3D> TiffReader::readTiffFile(const std::string& filename, int channel) {
    try {
        TIFFSetWarningHandler(customTifWarningHandler);
        ImageMetaData metaData = TiffReader::readMetadata_(filename);

        Image3D image;
        BoxCoord region{CuboidPosition{0, 0, 0}, CuboidShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};

        readSubimageFromTiffFileStatic(filename, metaData, region, image, channel);

        return image;

    } catch (const TiffMemoryException& e) {
        spdlog::warn("Insufficient memory to read TIFF file {}: {}", filename, e.what());
        return std::nullopt;
    } catch (const TiffException& e) {
        spdlog::error("{}", e.what());
        return std::nullopt;
    } catch (const std::runtime_error& e) {
        spdlog::error("{}",e.what());
        return std::nullopt;
    }
}

std::optional<ImageMetaData> TiffReader::readMetadata(const std::string& filename){
    try{
        return std::optional<ImageMetaData>(TiffReader::readMetadata_(filename));
    }
    catch(...){
        return std::nullopt;
    }
}

ImageMetaData TiffReader::readMetadata_(const std::string& filename) {
    TIFF* tifFile = TIFFOpen(filename.c_str(), "r");
    if (!tifFile) {
        throw TiffFileOpenException(filename);
    }

    try {
        ImageMetaData metaData = extractMetadataFromTiff(tifFile);
        metaData.filename = filename;
        TIFFClose(tifFile);

        if(metaData.slices < 1){
            if (metaData.totalImages != SIZE_MAX) metaData.slices = metaData.totalImages + 1;
        }

        return metaData;
    } catch (...) {
        TIFFClose(tifFile);
        throw;
    }
}


void TiffReader::readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel){

    TIFFSetWarningHandler(TiffReader::customTifWarningHandler);
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (!tif) {
        throw TiffFileOpenException(filename);
    }

    try {
        readSubimageFromTiffFile(tif, metaData, region, image, channel);
        TIFFClose(tif);
    } catch (const TiffException& e) {
        TIFFClose(tif);
        throw;
    } catch (const std::exception& e){
        TIFFClose(tif);
        throw TiffReadException(e.what());
    }
}


void TiffReader::resolveChannel(int channel, const ImageMetaData& metaData, int& ifdchannel, int& sppchannel) {
    if (channel < 0) {
        channel = 0;
    }
    ifdchannel = 0;
    sppchannel = 0;
    if (metaData.linChannels > 1) {
        ifdchannel = channel - 1;
        if (ifdchannel > metaData.linChannels - 1) {
            throw TiffMetadataException("Specified channel " + std::to_string(channel) +
                                    " larger than maximum number of image file directories: " +
                                    std::to_string(metaData.linChannels));
        }
    } else if (metaData.samplesPerPixel > 1) {
        sppchannel = channel - 1;
        if (sppchannel > metaData.samplesPerPixel - 1) {
            throw TiffMetadataException("Specified channel " + std::to_string(channel) +
                                    " larger than maximum number of samples per pixel: " +
                                    std::to_string(metaData.samplesPerPixel));
        }
    }
}


void TiffReader::readSubimageFromTiffFile(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel) {
    try {
        if (!tiffile) {
            throw TiffException("TIFF File is not open");
        }

        if (channel < 0) {
            spdlog::warn("Invalid channel {}, using channel 0", channel);
            channel = 0;
        }

        image = Image3D(region.dimensions, -1.0f);

        if (metaData.isTiled) {
            readTiledSubimage(tiffile, metaData, region, image, channel);
        } else if (metaData.rowsPerStrip > 0) {
            readStrippedSubimage(tiffile, metaData, region, image, channel);
        } else {
            readScanlineSubimage(tiffile, metaData, region, image, channel);
        }

        spdlog::info("Successfully read chunk ({}): ({},{},{}) {}x{}x{}", metaData.filename, region.position.width, region.position.height, region.position.depth, region.dimensions.width, region.dimensions.height, region.dimensions.depth);
    } catch (const TiffException& e) {
        throw e;
    } catch (const std::exception& e){
        throw TiffReadException(e.what());
    } catch (...){
        throw std::runtime_error("Unexpected error in tiffReader");
    }
}


void TiffReader::readScanlineSubimage(TIFF* tiffile, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel) {
    int ifdchannel, sppchannel;
    resolveChannel(channel, metaData, ifdchannel, sppchannel);

    tsize_t scanlineSize = TIFFScanlineSize(tiffile);
    std::vector<char> buf(scanlineSize);
    std::vector<float> rowData(region.dimensions.width);

    forEachZSlice(tiffile, region.position.depth, region.dimensions.depth, metaData, ifdchannel, [&](uint32_t zIndex) {
        for (uint32_t yIndex = static_cast<uint32_t>(region.position.height); yIndex < static_cast<uint32_t>(region.position.height + region.dimensions.height); yIndex++) {
            if (TIFFReadScanline(tiffile, buf.data(), yIndex) == -1) {
                throw TiffReadException("Failed to read scanline " + std::to_string(yIndex) +
                                        " in z-slice " + std::to_string(zIndex));
            }
            convertScanlineToFloat(buf.data(), rowData, region.dimensions.width, metaData, sppchannel);
            image.setRow(yIndex - region.position.height, zIndex - region.position.depth, rowData.data());
        }
    });
}


void TiffReader::readStrippedSubimage(TIFF* tif, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel) {
    int ifdchannel, sppchannel;
    resolveChannel(channel, metaData, ifdchannel, sppchannel);

    uint32_t rowsPerStrip = metaData.rowsPerStrip;
    tsize_t stripSize = TIFFStripSize(tif);
    tsize_t scanlineSize = TIFFScanlineSize(tif);

    std::vector<char> stripBuf(stripSize);
    std::vector<float> rowData(region.dimensions.width);

    forEachZSlice(tif, region.position.depth, region.dimensions.depth, metaData, ifdchannel, [&](uint32_t zIndex) {
        uint32_t startStrip = static_cast<uint32_t>(region.position.height) / rowsPerStrip;
        uint32_t endStrip = (static_cast<uint32_t>(region.position.height) + static_cast<uint32_t>(region.dimensions.height) - 1) / rowsPerStrip;

        for (uint32_t s = startStrip; s <= endStrip; s++) {
            tmsize_t bytesRead = TIFFReadEncodedStrip(tif, s, stripBuf.data(), stripSize);
            if (bytesRead == -1) {
                throw TiffReadException("Failed to read strip " + std::to_string(s) +
                                      " in z-slice " + std::to_string(zIndex));
            }

            uint32_t stripStartRow = s * rowsPerStrip;
            uint32_t stripEndRow = std::min((s + 1) * rowsPerStrip, static_cast<uint32_t>(metaData.imageLength));

            uint32_t rowStart = std::max(static_cast<uint32_t>(region.position.height), stripStartRow);
            uint32_t rowEnd = std::min(static_cast<uint32_t>(region.position.height + region.dimensions.height), stripEndRow);

            for (uint32_t row = rowStart; row < rowEnd; row++) {
                uint32_t rowInStrip = row - stripStartRow;
                const char* scanlineData = stripBuf.data() + rowInStrip * scanlineSize;

                convertScanlineToFloat(scanlineData, rowData, region.dimensions.width, metaData, sppchannel);
                image.setRow(row - region.position.height, zIndex - region.position.depth, rowData.data());
            }
        }
    });
}


void TiffReader::readTiledSubimage(TIFF* tif, const ImageMetaData& metaData, const BoxCoord& region, Image3D& image, int channel) {
    int ifdchannel, sppchannel;
    resolveChannel(channel, metaData, ifdchannel, sppchannel);

    uint32_t tileWidth = metaData.tileWidth;
    uint32_t tileLength = metaData.tileLength;
    tsize_t tileSize = TIFFTileSize(tif);
    tsize_t tileRowSize = TIFFTileRowSize(tif);

    uint32_t tilesPerRow = (static_cast<uint32_t>(metaData.imageWidth) + tileWidth - 1) / tileWidth;
    uint32_t startTileCol = static_cast<uint32_t>(region.position.width) / tileWidth;
    uint32_t endTileCol = static_cast<uint32_t>(region.position.width + region.dimensions.width - 1) / tileWidth;
    uint32_t numTilesToRead = endTileCol - startTileCol + 1;

    std::vector<float> rowData(region.dimensions.width);

    forEachZSlice(tif, region.position.depth, region.dimensions.depth, metaData, ifdchannel, [&](uint32_t zIndex) {
        uint32_t startTileRow = static_cast<uint32_t>(region.position.height) / tileLength;
        uint32_t endTileRow = (static_cast<uint32_t>(region.position.height + region.dimensions.height - 1)) / tileLength;

        for (uint32_t tr = startTileRow; tr <= endTileRow; tr++) {
            uint32_t tileRowStart = tr * tileLength;
            uint32_t tileRowEnd = std::min((tr + 1) * tileLength, static_cast<uint32_t>(metaData.imageLength));

            uint32_t rowStart = std::max(static_cast<uint32_t>(region.position.height), tileRowStart);
            uint32_t rowEnd = std::min(static_cast<uint32_t>(region.position.height + region.dimensions.height), tileRowEnd);

            std::vector<std::vector<char>> tileCache(numTilesToRead);
            for (uint32_t i = 0; i < numTilesToRead; i++) {
                uint32_t tileIndex = tr * tilesPerRow + (startTileCol + i);
                tileCache[i].resize(tileSize);
                if (TIFFReadEncodedTile(tif, tileIndex, tileCache[i].data(), tileSize) == -1) {
                    throw TiffReadException("Failed to read tile " + std::to_string(tileIndex) +
                                          " in z-slice " + std::to_string(zIndex));
                }
            }

            for (uint32_t row = rowStart; row < rowEnd; row++) {
                std::fill(rowData.begin(), rowData.end(), -1.0f);
                uint32_t rowInTile = row - tileRowStart;

                for (uint32_t i = 0; i < numTilesToRead; i++) {
                    uint32_t tc = startTileCol + i;
                    uint32_t tileColStart = tc * tileWidth;
                    uint32_t tileColEnd = std::min((tc + 1) * tileWidth, static_cast<uint32_t>(metaData.imageWidth));
                    uint32_t tilePixelWidth = tileColEnd - tileColStart;

                    size_t destOffset = tileColStart - static_cast<uint32_t>(region.position.width);
                    const char* tileRowData = tileCache[i].data() + rowInTile * tileRowSize;
                    convertTileRowToFloat(tileRowData, rowData, destOffset, tilePixelWidth, metaData, sppchannel);
                }

                image.setRow(row - region.position.height, zIndex - region.position.depth, rowData.data());
            }
        }
    });
}


BoxCoordWithPadding TiffReader::computeReadSource(const BoxCoordWithPadding& box) const {
    BoxCoord image{CuboidShape{0,0,0}, CuboidShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
    BoxCoord paddedBox = box.getBox();
    Padding padding = paddedBox.cropTo(image);
    padding.before.width = box.padding.before.width;
    padding.after.width = box.padding.after.width;

    if (metaData.isTiled) {
        uint32_t tw = metaData.tileWidth;
        uint32_t tl = metaData.tileLength;

        int64_t startX = std::max<int64_t>(0, paddedBox.position.width);
        int64_t endX = std::min<int64_t>(metaData.imageWidth, paddedBox.position.width + static_cast<int64_t>(paddedBox.dimensions.width));
        int64_t startY = std::max<int64_t>(0, paddedBox.position.height);
        int64_t endY = std::min<int64_t>(metaData.imageLength, paddedBox.position.height + static_cast<int64_t>(paddedBox.dimensions.height));

        uint32_t tileStartX = static_cast<uint32_t>(startX) / tw * tw;
        uint32_t tileEndX = std::min(static_cast<uint32_t>(metaData.imageWidth), (static_cast<uint32_t>(endX - 1) / tw + 1) * tw);
        uint32_t tileStartY = static_cast<uint32_t>(startY) / tl * tl;
        uint32_t tileEndY = std::min(static_cast<uint32_t>(metaData.imageLength), (static_cast<uint32_t>(endY - 1) / tl + 1) * tl);

        BoxCoord sourceBox{
            CuboidPosition{static_cast<int64_t>(tileStartX), static_cast<int64_t>(tileStartY), paddedBox.position.depth},
            CuboidShape{tileEndX - tileStartX, tileEndY - tileStartY, paddedBox.dimensions.depth}
        };
        return BoxCoordWithPadding{sourceBox, padding};
    } else {
        uint32_t rps = metaData.rowsPerStrip > 0 ? metaData.rowsPerStrip : static_cast<uint32_t>(metaData.imageLength);

        int64_t startY = std::max<int64_t>(0, paddedBox.position.height);
        int64_t endY = std::min<int64_t>(metaData.imageLength, paddedBox.position.height + static_cast<int64_t>(paddedBox.dimensions.height));

        uint32_t stripStartY = static_cast<uint32_t>(startY) / rps * rps;
        uint32_t stripEndY = std::min(static_cast<uint32_t>(metaData.imageLength), (static_cast<uint32_t>(endY - 1) / rps + 1) * rps);

        BoxCoord sourceBox{
            CuboidPosition{0, static_cast<int64_t>(stripStartY), paddedBox.position.depth},
            CuboidShape{metaData.imageWidth, stripEndY - stripStartY, paddedBox.dimensions.depth}
        };
        return BoxCoordWithPadding{sourceBox, padding};
    }
}


PaddedImage TiffReader::extractFromBuffer(const BoxCoordWithPadding& coord, BufferEntry& entry) const {
    BoxCoord convertedCoords{
        coord.box.position - entry.source.box.position - coord.padding.before + entry.source.padding.before,
        coord.box.dimensions + coord.padding.before + coord.padding.after
    };
    PaddedImage result;
    result.image = entry.image.getSubimageCopy(convertedCoords);
    result.padding = coord.padding;
    return result;
}


void TiffReader::executeRead(std::list<PendingRead>::iterator pendingIt, const BoxCoordWithPadding& source) const {
    Image3D readImage;
    try {
        auto guard = handlePool_->acquire();
        readSubimageFromTiffFile(guard.get(), metaData, source.box, readImage, channel);
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [box, promise] : pendingIt->waiters) {
            promise->set_exception(std::current_exception());
        }
        pendingReads_.erase(pendingIt);
        inFlightReads_.fetch_sub(1);
        prefetchCv_.notify_all();
        return;
    }

    ImagePadding::padImage(readImage, source.padding, PaddingFillType::MIRROR);

    std::vector<std::pair<BoxCoordWithPadding, std::shared_ptr<std::promise<PaddedImage>>>> waiters;
    std::list<BufferEntry>::iterator bufferIt;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        waiters = std::move(pendingIt->waiters);
        pendingReads_.erase(pendingIt);

        bufferIt = bufferedRegions_.emplace(bufferedRegions_.end());
        bufferIt->image = std::move(readImage);
        bufferIt->source = source;

        inFlightReads_.fetch_sub(1);
        prefetchCv_.notify_all();
    }

    for (auto& [box, promise] : waiters) {
        try {
            promise->set_value(extractFromBuffer(box, *bufferIt));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }
}


std::future<PaddedImage> TiffReader::getSubimage(const BoxCoordWithPadding& box) const {
    if (config_.prefetchCount > 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        prefetchCv_.wait(lock, [this] { return inFlightReads_.load() < config_.prefetchCount; });
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // check if region is already in some buffer
    for (auto& entry : bufferedRegions_) {
        if (box.isWithin(entry.source)) {
            std::promise<PaddedImage> p;
            p.set_value(extractFromBuffer(box, entry));
            return p.get_future();
        }
    }

    // check if reading that region is already queued, wait for result
    for (auto it = pendingReads_.begin(); it != pendingReads_.end(); ++it) {
        if (box.isWithin(it->source)) {
            auto promise = std::make_shared<std::promise<PaddedImage>>();
            auto future = promise->get_future();
            it->waiters.push_back({box, promise});
            return future;
        }
    }

    // queue region reading task
    BoxCoordWithPadding source = computeReadSource(box);
    auto promise = std::make_shared<std::promise<PaddedImage>>();
    auto future = promise->get_future();

    auto& pending = pendingReads_.emplace_back();
    pending.source = source;
    pending.waiters.push_back({box, promise});

    auto pendingIt = std::prev(pendingReads_.end());
    inFlightReads_.fetch_add(1);

    readerPool_->enqueue([this, source, pendingIt]() {
        executeRead(pendingIt, source);
    });

    return future;
}


void TiffReader::prefetch(const std::vector<BoxCoordWithPadding>& boxes) const {
    if (!config_.prefetchEnabled) return;

    for (const auto& box : boxes) {
        if (config_.prefetchCount > 0) {
            std::unique_lock<std::mutex> waitLock(mutex_);
            prefetchCv_.wait(waitLock, [this] { return inFlightReads_.load() < config_.prefetchCount; });
        }

        std::lock_guard<std::mutex> lock(mutex_);

        bool found = false;
        for (const auto& entry : bufferedRegions_) {
            if (box.isWithin(entry.source)) {
                found = true;
                break;
            }
        }
        if (found) continue;

        for (const auto& pending : pendingReads_) {
            if (box.isWithin(pending.source)) {
                found = true;
                break;
            }
        }
        if (found) continue;

        BoxCoordWithPadding source = computeReadSource(box);
        inFlightReads_.fetch_add(1);

        auto& pending = pendingReads_.emplace_back();
        pending.source = source;

        auto pendingIt = std::prev(pendingReads_.end());

        readerPool_->enqueue([this, source, pendingIt]() {
            executeRead(pendingIt, source);
        });
    }
}


const ImageMetaData& TiffReader::getMetaData() const {
    return metaData;
}

void TiffReader::convertImageTo32F(Image3D& image, const ImageMetaData& metaData){
    CuboidShape shape = image.getShape();
    double scale = 1.0 / (metaData.maxSampleValue - metaData.minSampleValue);
    double offset = -metaData.minSampleValue * scale;

    for (auto it = image.begin(); it != image.end(); ++it) {
        float originalValue = *it;
        float convertedValue = static_cast<float>(originalValue * scale + offset);
        *it = convertedValue;
    }
}


ImageMetaData TiffReader::extractMetadataFromTiff(TIFF*& tifFile) {
    ImageMetaData metadatatemp;

    char* img_description = nullptr;
    if (TIFFGetField(tifFile, TIFFTAG_IMAGEDESCRIPTION, &img_description)) {
        metadatatemp.description = img_description;

        std::istringstream iss(metadatatemp.description);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find("channels=") != std::string::npos)
                metadatatemp.linChannels = std::stoi(line.substr(line.find("=") + 1));
            else if (line.find("slices=") != std::string::npos)
                metadatatemp.slices = std::stoi(line.substr(line.find("=") + 1));
        }
    }
    if (metadatatemp.slices == 0) {
        int totalDirectories = countTiffDirectories(tifFile);
        metadatatemp.slices = totalDirectories / metadatatemp.linChannels;
    }

    uint32_t width = 0, length = 0;
    uint16_t spp = 1, bps = 0, photo = 0, planar = 0;
    uint16_t sampleFormat = SAMPLEFORMAT_UINT;
    float xres = 0.f, yres = 0.f;
    uint16_t resUnit = RESUNIT_NONE;

    TIFFGetField(tifFile, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tifFile, TIFFTAG_IMAGELENGTH, &length);
    TIFFGetField(tifFile, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetField(tifFile, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tifFile, TIFFTAG_PHOTOMETRIC, &photo);
    TIFFGetField(tifFile, TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetField(tifFile, TIFFTAG_RESOLUTIONUNIT, &resUnit);
    TIFFGetField(tifFile, TIFFTAG_XRESOLUTION, &xres);
    TIFFGetField(tifFile, TIFFTAG_YRESOLUTION, &yres);

    if (!TIFFGetField(tifFile, TIFFTAG_SAMPLEFORMAT, &sampleFormat)) {
        sampleFormat = SAMPLEFORMAT_UINT;
    }

    metadatatemp.imageWidth  = width;
    metadatatemp.imageLength = length;
    metadatatemp.samplesPerPixel = spp;
    metadatatemp.bitsPerSample = bps;
    metadatatemp.photometricInterpretation = photo;
    metadatatemp.planarConfig = planar;
    metadatatemp.sampleFormat = sampleFormat;
    metadatatemp.resolutionUnit = resUnit;
    metadatatemp.xResolution = xres;
    metadatatemp.yResolution = yres;

    if (TIFFIsTiled(tifFile)) {
        metadatatemp.isTiled = true;
        TIFFGetField(tifFile, TIFFTAG_TILEWIDTH, &metadatatemp.tileWidth);
        TIFFGetField(tifFile, TIFFTAG_TILELENGTH, &metadatatemp.tileLength);
        metadatatemp.rowsPerStrip = 0;
    } else {
        metadatatemp.isTiled = false;
        metadatatemp.tileWidth = 0;
        metadatatemp.tileLength = 0;
        uint32_t rps = 0;
        if (TIFFGetField(tifFile, TIFFTAG_ROWSPERSTRIP, &rps) && rps > 0) {
            metadatatemp.rowsPerStrip = rps;
        } else {
            metadatatemp.rowsPerStrip = static_cast<uint32_t>(length);
        }
    }

    uint16_t minTag = 0, maxTag = 0;
    bool hasMinTag = TIFFGetField(tifFile, TIFFTAG_MINSAMPLEVALUE, &minTag);
    bool hasMaxTag = TIFFGetField(tifFile, TIFFTAG_MAXSAMPLEVALUE, &maxTag);

    if (hasMinTag && hasMaxTag) {
        metadatatemp.minSampleValue = minTag;
        metadatatemp.maxSampleValue = maxTag;
    }

    if (sampleFormat == SAMPLEFORMAT_UINT) {
        metadatatemp.minSampleValue = 0.0;
        metadatatemp.maxSampleValue = std::pow(2.0, bps) - 1.0;
    }
    else if (sampleFormat == SAMPLEFORMAT_INT) {
        metadatatemp.minSampleValue = -std::pow(2.0, bps - 1);
        metadatatemp.maxSampleValue =  std::pow(2.0, bps - 1) - 1.0;
    }
    else if (sampleFormat == SAMPLEFORMAT_IEEEFP) {
        metadatatemp.minSampleValue = 0.0;
        metadatatemp.maxSampleValue = 1.0;
    }

    return metadatatemp;
}


int TiffReader::countTiffDirectories(TIFF* tif) {
    int count = 0;
    do {
        count++;
    } while (TIFFReadDirectory(tif));

    TIFFSetDirectory(tif, 0);
    return count;
}


void TiffReader::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    auto logger = spdlog::get("reader");
    if (!logger) {
        return;
    }

    va_list ap_copy;
    va_copy(ap_copy, ap);
    int required = vsnprintf(nullptr, 0, fmt, ap_copy);
    va_end(ap_copy);

    if (required < 0) {
        logger->debug("TIFF warning (format error): {}", fmt);
        return;
    }

    std::string message;
    message.resize(static_cast<size_t>(required));
    vsnprintf(&message[0], static_cast<size_t>(required) + 1, fmt, ap);

    logger->debug("Tiff Warning Handler: {}", message);
}


void TiffReader::convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, size_t width, const ImageMetaData& metaData, int channel) {
    const uint16_t spp = metaData.samplesPerPixel;

    if (metaData.bitsPerSample == 8) {
        const uint8_t* data8 = reinterpret_cast<const uint8_t*>(scanlineData);
        for (size_t x = 0; x < width; x++) {
            rowData[x] = static_cast<float>(data8[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 16) {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(scanlineData);
        for (size_t x = 0; x < width; x++) {
            rowData[x] = static_cast<float>(data16[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 32) {
        const float* data32 = reinterpret_cast<const float*>(scanlineData);
        for (size_t x = 0; x < width; x++) {
            rowData[x] = data32[x * spp + channel];
        }
    } else {
        throw TiffReadException("Unsupported bit depth: " + std::to_string(metaData.bitsPerSample) +
                                " (supported: 8, 16, 32)");
    }
}


void TiffReader::convertTileRowToFloat(const char* tileRowData, std::vector<float>& rowData, size_t xOffset, size_t tilePixelWidth, const ImageMetaData& metaData, int channel) {
    const uint16_t spp = metaData.samplesPerPixel;

    if (metaData.bitsPerSample == 8) {
        const uint8_t* data8 = reinterpret_cast<const uint8_t*>(tileRowData);
        for (size_t x = 0; x < tilePixelWidth; x++) {
            rowData[xOffset + x] = static_cast<float>(data8[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 16) {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(tileRowData);
        for (size_t x = 0; x < tilePixelWidth; x++) {
            rowData[xOffset + x] = static_cast<float>(data16[x * spp + channel]);
        }
    } else if (metaData.bitsPerSample == 32) {
        const float* data32 = reinterpret_cast<const float*>(tileRowData);
        for (size_t x = 0; x < tilePixelWidth; x++) {
            rowData[xOffset + x] = data32[x * spp + channel];
        }
    } else {
        throw TiffReadException("Unsupported bit depth: " + std::to_string(metaData.bitsPerSample) +
                                " (supported: 8, 16, 32)");
    }
}
