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

#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/ImageOperations.h"
#include <cstdint>
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

uint16_t WriterCompressionConfig::parseCompression(const std::string& s) {
    if (s == "none" || s == "NONE" || s == "1") return COMPRESSION_NONE;
    if (s == "lzw" || s == "LZW" || s == "5") return COMPRESSION_LZW;
    if (s == "deflate" || s == "DEFLATE" || s == "zip" || s == "ZIP" || s == "8") return COMPRESSION_DEFLATE;
    spdlog::warn("Unknown compression scheme '{}', defaulting to none", s);
    return COMPRESSION_NONE;
}

const char* WriterCompressionConfig::compressionToString(uint16_t scheme) {
    switch (scheme) {
        case COMPRESSION_NONE: return "none";
        case COMPRESSION_LZW: return "lzw";
        case COMPRESSION_DEFLATE: return "deflate";
        default: return "unknown";
    }
}

// ============================================================================
// ITiffRegionWriter implementations
// ============================================================================

void TiffRegionWriterStripped::writeSlice(TIFF* tif, const float* sliceData, const ImageMetaData& metaData) const {
    tmsize_t stripSize = TIFFStripSize(tif);
    tsize_t scanlineSize = TIFFScanlineSize(tif);

    if (stripSize <= 0) {
        throw TiffWriteException("Invalid strip size: " + std::to_string(stripSize));
    }

    char* buf = static_cast<char*>(_TIFFmalloc(stripSize));
    if (!buf) {
        throw TiffMemoryException("Memory allocation failed for strip buffer");
    }

    tmsize_t totalBytes = static_cast<tmsize_t>(scanlineSize) * static_cast<tmsize_t>(metaData.imageLength);

    if (totalBytes <= stripSize) {
        memcpy(buf, sliceData, totalBytes);
        if (TIFFWriteEncodedStrip(tif, 0, buf, totalBytes) == -1) {
            _TIFFfree(buf);
            throw TiffWriteException("Failed to write encoded strip");
        }
    } else {
        uint32_t rowsPerStrip = 0;
        TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = static_cast<uint32_t>(metaData.imageLength);

        uint32_t stripCount = (static_cast<uint32_t>(metaData.imageLength) + rowsPerStrip - 1) / rowsPerStrip;
        tmsize_t offset = 0;

        for (uint32_t s = 0; s < stripCount; ++s) {
            uint32_t rowsInThisStrip = std::min(rowsPerStrip, static_cast<uint32_t>(metaData.imageLength) - s * rowsPerStrip);
            tmsize_t bytesInStrip = static_cast<tmsize_t>(scanlineSize) * rowsInThisStrip;

            memcpy(buf, reinterpret_cast<const char*>(sliceData) + offset, bytesInStrip);
            if (TIFFWriteEncodedStrip(tif, s, buf, bytesInStrip) == -1) {
                _TIFFfree(buf);
                throw TiffWriteException("Failed to write encoded strip " + std::to_string(s));
            }
            offset += bytesInStrip;
        }
    }

    _TIFFfree(buf);
}


TiffRegionWriterTiled::TiffRegionWriterTiled(uint32_t tileWidth, uint32_t tileLength)
    : tileWidth_(tileWidth), tileLength_(tileLength) {}

void TiffRegionWriterTiled::writeSlice(TIFF* tif, const float* sliceData, const ImageMetaData& metaData) const {
    tmsize_t tileSize = TIFFTileSize(tif);
    if (tileSize <= 0) {
        throw TiffWriteException("Invalid tile size: " + std::to_string(tileSize));
    }

    uint32_t width = static_cast<uint32_t>(metaData.imageWidth);
    uint32_t height = static_cast<uint32_t>(metaData.imageLength);
    uint32_t tilesPerRow = (width + tileWidth_ - 1) / tileWidth_;
    uint32_t tilesPerCol = (height + tileLength_ - 1) / tileLength_;

    std::vector<char> buf(tileSize);

    for (uint32_t tr = 0; tr < tilesPerCol; ++tr) {
        for (uint32_t tc = 0; tc < tilesPerRow; ++tc) {
            uint32_t tileIndex = tr * tilesPerRow + tc;

            std::fill(buf.begin(), buf.end(), 0);

            uint32_t tileRowStart = tr * tileLength_;
            uint32_t tileColStart = tc * tileWidth_;
            uint32_t rowsInTile = std::min(tileLength_, height - tileRowStart);
            uint32_t colsInTile = std::min(tileWidth_, width - tileColStart);

            float* tileData = reinterpret_cast<float*>(buf.data());
            for (uint32_t r = 0; r < rowsInTile; ++r) {
                const float* srcRow = sliceData + (tileRowStart + r) * width + tileColStart;
                std::memcpy(tileData + r * tileWidth_, srcRow, colsInTile * sizeof(float));
            }

            if (TIFFWriteEncodedTile(tif, tileIndex, buf.data(), tileSize) == -1) {
                throw TiffWriteException("Failed to write encoded tile " + std::to_string(tileIndex));
            }
        }
    }
}

// ============================================================================
// TiffWriter
// ============================================================================

TiffWriter::TiffWriter(const std::string& filename, const CuboidShape& imageShape)
    : outputFilename(filename),
    imageShape(imageShape)
    {
    TIFFSetWarningHandler(TiffWriter::customTifWarningHandler);
    try {
        this->tif = openTiff(filename.c_str(), imageShape);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        throw;
    }
    regionWriter_ = std::make_unique<TiffRegionWriterStripped>();
}


void TiffWriter::configure(WriterCompressionConfig compressionConfig, WriterConfig writerConfig){
    this->compressionConfig = compressionConfig;
    this->writerConfig_ = writerConfig;

    if (writerConfig_.useTiles()) {
        regionWriter_ = std::make_unique<TiffRegionWriterTiled>(writerConfig_.tileWidth, writerConfig_.tileLength);
    } else {
        regionWriter_ = std::make_unique<TiffRegionWriterStripped>();
    }

    spdlog::debug("TiffWriter configured: compression={}, level={}, tiles={}x{}",
        WriterCompressionConfig::compressionToString(compressionConfig.compressionScheme),
        compressionConfig.compressionLevel,
        writerConfig.tileWidth, writerConfig.tileLength);
}


TiffWriter::~TiffWriter() {
    assert(tileBuffer.size() == 0 && readyTiles_.empty() && "Buffers not empty but done writing, should not happen");
    if (tif) {
        TIFFClose(tif);
    }
}


bool TiffWriter::setSubimage(const Image3D& image, const BoxCoord& coord,
                             const CuboidPosition& sourceOffset) const {
    CuboidShape imgShape = image.getShape();
    assert(imgShape.depth != 0 || imgShape.width != 0 || imgShape.height != 0 && "Cannot set subimage: Image3D has invalid dimensions");

    std::unique_lock<std::mutex> lock(writerMutex);

    int bufferIndex;
    bufferIndex = getStripIndex(coord);
    try{
        if (bufferIndex != -1){
            copyToTile(image, coord, sourceOffset, bufferIndex);
        }
        else{
            createNewTile(coord);

            bufferIndex = getStripIndex(coord);
            copyToTile(image, coord, sourceOffset, bufferIndex);
        }
        return true;
    } catch (const TiffException& e) {
        spdlog::error("TIFF error in setSubimage {}", e.what());
        return false;
    } catch (const std::exception& e) {
        spdlog::error("Exception in setSubimage {}", e.what());
        return false;
    } catch (...) {
        spdlog::error("Unknown exception in setSubimage");
        return false;
    }

}


void TiffWriter::createNewTile(const BoxCoord& coord) const {
    ImageBuffer tile;
    tile.source = BoxCoord{
        CuboidPosition{0, 0, coord.position.depth},
        CuboidShape{imageShape.width, imageShape.height, coord.dimensions.depth}
    };
    tile.remainingRegions = {tile.source};

    Image3D image(tile.source.dimensions, -1.0f);
    tile.image = std::move(image);

    tileBuffer.push_back(std::move(tile));
}


int TiffWriter::getStripIndex(const BoxCoord& coord) const {
    for (size_t i = 0; i < tileBuffer.size(); i++){
        if (coord.isWithin(tileBuffer.find(i).source)){
            return static_cast<int>(i);
        }
    }
    return -1;
}


void TiffWriter::copyToTile(const Image3D& image, const BoxCoord& coord, const CuboidPosition& sourceOffset, int index) const {
    ImageBuffer& tile = tileBuffer.find(index);
    BoxCoord srcBox = coord;
    srcBox.position = srcBox.position - tile.source.position;

    BoxCoord sourceRegion{sourceOffset, coord.dimensions};
    ImageOperations::insertCubeInImage(image, sourceRegion, tile.image, srcBox);

    std::vector<BoxCoord> newRemaining;
    newRemaining.reserve(tile.remainingRegions.size() * 6);
    for (const auto& region : tile.remainingRegions) {
        auto fragments = subtractBox(region, coord);
        for (auto& frag : fragments) {
            if (frag.dimensions.getVolume() > 0)
                newRemaining.push_back(std::move(frag));
        }
    }
    tile.remainingRegions = std::move(newRemaining);

    if (isTileFull(tile)) {
        int64_t zPos = tile.source.position.depth;
        ImageBuffer tileToMove = std::move(tile);
        tileBuffer.deleteIndex(index);
        readyTiles_[zPos] = std::move(tileToMove);
        processReadyToWriteQueue();
    }
}


bool TiffWriter::isTileFull(const ImageBuffer& strip) const {
    return strip.remainingRegions.empty();
}


void TiffWriter::processReadyToWriteQueue() const {
    while (!readyTiles_.empty()) {
        auto it = readyTiles_.begin();
        if (it->first != static_cast<int64_t>(writtenToDepth)) {
            break;
        }

        ImageBuffer tile = std::move(it->second);
        readyTiles_.erase(it);

        writeTile(tile);
    }
}


void TiffWriter::writeTile(const ImageBuffer& tile) const {
    size_t z = tile.source.position.depth;
    size_t depth = tile.source.dimensions.depth;
    writeToFile_(z, depth, tile.image);
}


bool TiffWriter::writeToFile_(size_t z, size_t depth, const Image3D& layers) const {
    if (!tif) {
        throw TiffWriteException("TIFF file handle is null");
    }

    ImageMetaData metaData = extractMetaData(layers);
    size_t slicePixels = metaData.imageWidth * metaData.imageLength;
    const float* buffer = layers.getItkImage()->GetBufferPointer();

    for (size_t i = 0; i < depth; ++i) {
        setTiffFields(tif, metaData, compressionConfig, writerConfig_);
        regionWriter_->writeSlice(tif, buffer + (z - writtenToDepth + i) * slicePixels, metaData);
        if (!TIFFWriteDirectory(tif)) {
            throw TiffWriteException("Failed to set directory for slice " + std::to_string(z + i));
        }
    }

    writtenToDepth = z + depth;
    spdlog::info("Successfully saved ImageFileDirectory ({}): {} - {}", outputFilename, z, z + depth);
    return true;
}


void TiffWriter::setTiffFields(TIFF* tif, const ImageMetaData& metaData,
                              const WriterCompressionConfig& compression,
                              const WriterConfig& writerConfig){
    try {
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(metaData.imageWidth));
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(metaData.imageLength));
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, metaData.bitsPerSample);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, metaData.samplesPerPixel);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, metaData.planarConfig);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, metaData.photometricInterpretation);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, metaData.sampleFormat);

        if (writerConfig.useTiles()) {
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, writerConfig.tileWidth);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, writerConfig.tileLength);
        } else {
            uint32_t rowsPerStrip = static_cast<uint32_t>(metaData.imageLength);
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);
        }

        TIFFSetField(tif, TIFFTAG_COMPRESSION, compression.compressionScheme);

        if (compression.compressionScheme == COMPRESSION_LZW || compression.compressionScheme == COMPRESSION_DEFLATE) {
            if (metaData.sampleFormat == SAMPLEFORMAT_IEEEFP) {
                TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_NONE);
            } else {
                TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
            }
        }

        if (compression.compressionScheme == COMPRESSION_DEFLATE && compression.compressionLevel >= 0) {
            TIFFSetField(tif, TIFFTAG_ZIPQUALITY, compression.compressionLevel);
        }
    } catch (const std::exception& e) {
        throw TiffMetadataException("Failed to set TIFF fields: " + std::string(e.what()));
    } catch (...) {
        throw TiffMetadataException("Unknown error while setting TIFF fields");
    }
}


void TiffWriter::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    auto logger = spdlog::get("writer");
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


ImageMetaData TiffWriter::extractMetaData(const Image3D& image){
    ImageMetaData metaData;
    CuboidShape size = image.getShape();
    metaData.imageWidth = size.width;
    metaData.imageLength = size.height;
    metaData.slices = size.depth;
    metaData.bitsPerSample = 32;
    metaData.samplesPerPixel = 1;
    metaData.sampleFormat = SAMPLEFORMAT_IEEEFP;
    metaData.photometricInterpretation = 1;
    metaData.planarConfig = 1;
    return metaData;
}


TIFF* TiffWriter::openTiff(const char* filename, const CuboidShape& size){
    uint64_t imageBytes = size.getVolume() * 4;

    const uint64_t TWO_GIGABYTES = 2ULL * 1024 * 1024 * 1024;

    const char* mode = (imageBytes >= TWO_GIGABYTES) ? "w8" : "w";
    TIFF* tif = TIFFOpen(filename, mode);
    if (!tif) {
        throw TiffFileOpenException(filename);
    }
    return tif;
}


bool TiffWriter::writeToFile(const std::string& filename, const Image3D& image,
                             WriterCompressionConfig compressionConfig,
                             WriterConfig writerConfig) {
    try {
        ImageMetaData metaData = extractMetaData(image);
        CuboidShape imgShape = image.getShape();

        if (imgShape.depth == 0 || imgShape.width == 0 || imgShape.height == 0) {
            // throw TiffWriteException("Cannot write Image3D: Invalid image dimensions");
        }

        TIFFSetWarningHandler(TiffWriter::customTifWarningHandler);
        TIFF* tif = openTiff(filename.c_str(), imgShape);

        spdlog::debug("Writing TIFF: compression={}, level={}, tiles={}x{}",
            WriterCompressionConfig::compressionToString(compressionConfig.compressionScheme),
            compressionConfig.compressionLevel,
            writerConfig.tileWidth, writerConfig.tileLength);

        std::unique_ptr<ITiffRegionWriter> regionWriter;
        if (writerConfig.useTiles()) {
            regionWriter = std::make_unique<TiffRegionWriterTiled>(writerConfig.tileWidth, writerConfig.tileLength);
        } else {
            regionWriter = std::make_unique<TiffRegionWriterStripped>();
        }

        size_t depth = imgShape.depth;
        size_t slicePixels = imgShape.width * imgShape.height;
        const float* buffer = image.getItkImage()->GetBufferPointer();

        for (size_t i = 0; i < depth; ++i) {
            setTiffFields(tif, metaData, compressionConfig, writerConfig);
            regionWriter->writeSlice(tif, buffer + i * slicePixels, metaData);
            if (!TIFFWriteDirectory(tif)) {
                throw TiffWriteException("Failed to set directory for slice " + std::to_string(i));
            }
        }

        TIFFClose(tif);

        return true;

    } catch (const std::exception& e) {
        spdlog::error("Exception in writeToFile: {}", e.what());
        return false;
    } catch (...) {
        spdlog::error("Unknown exception in writeToFile");
        return false;
    }
}


int TiffWriter::getTargetItkType(const ImageMetaData& metadata) {
    if (metadata.bitsPerSample == 8) {
        return 8;
    } else if (metadata.bitsPerSample == 16) {
        return 16;
    } else if (metadata.bitsPerSample == 32) {
        return 32;
    } else {
        throw TiffMetadataException("Unsupported bit depth from metadata: " + std::to_string(metadata.bitsPerSample));
    }
}


void TiffWriter::convertSliceDataToTargetType(const std::vector<float>& sourceData,
                                            std::vector<uint8_t>& targetData,
                                            size_t width, size_t height,
                                            const ImageMetaData& metadata) {
    size_t numPixels = width * height;

    if (metadata.bitsPerSample == 8) {
        targetData.resize(numPixels);
        for (size_t i = 0; i < numPixels; ++i) {
            float scaledValue = sourceData[i] * 255.0f;
            targetData[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, scaledValue)));
        }
    } else if (metadata.bitsPerSample == 16) {
        targetData.resize(numPixels * 2);
        uint16_t* data16 = reinterpret_cast<uint16_t*>(targetData.data());
        for (size_t i = 0; i < numPixels; ++i) {
            float scaledValue = sourceData[i] * 65535.0f;
            data16[i] = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, scaledValue)));
        }
    } else if (metadata.bitsPerSample == 32) {
        targetData.resize(numPixels * 4);
        float* data32 = reinterpret_cast<float*>(targetData.data());
        for (size_t i = 0; i < numPixels; ++i) {
            data32[i] = sourceData[i];
        }
    } else {
        throw TiffMetadataException("Unsupported bit depth: " + std::to_string(metadata.bitsPerSample));
    }
}
