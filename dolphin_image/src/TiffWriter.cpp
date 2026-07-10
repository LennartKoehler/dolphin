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
#include <climits>
#include <queue>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <itkMinimumMaximumImageFilter.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

uint16_t TiffCompressionConfig::parseCompression(const std::string& s) {
    if (s == "none" || s == "NONE" || s == "1") return COMPRESSION_NONE;
    if (s == "lzw" || s == "LZW" || s == "5") return COMPRESSION_LZW;
    if (s == "deflate" || s == "DEFLATE" || s == "zip" || s == "ZIP" || s == "8") return COMPRESSION_DEFLATE;
    spdlog::warn("Unknown compression scheme '{}', defaulting to none", s);
    return COMPRESSION_NONE;
}

const char* TiffCompressionConfig::compressionToString(uint16_t scheme) {
    switch (scheme) {
        case COMPRESSION_NONE: return "none";
        case COMPRESSION_LZW: return "lzw";
        case COMPRESSION_DEFLATE: return "deflate";
        default: return "unknown";
    }
}

// Constructor
TiffWriter::TiffWriter(const std::string& filename, const CuboidShape& imageShape, TiffCompressionConfig config)
    : outputFilename(filename),
    imageShape(imageShape),
    compressionConfig(config) {

    TIFFSetWarningHandler(TiffWriter::customTifWarningHandler);
    this->tif = openTiff(filename.c_str(), imageShape);
    spdlog::debug("TiffWriter created: compression={}, level={}",
        TiffCompressionConfig::compressionToString(compressionConfig.compressionScheme), compressionConfig.compressionLevel);
}



// Destructor
TiffWriter::~TiffWriter() {
    assert (tileBuffer.size() == 0 && "TileBuffer not empty but done writing, should not happen");
    if (tif) {
        TIFFClose(tif);
    }
}

// const ImageMetaData& TiffWriter::getMetaData() const {
//     return metaData;
// }

bool TiffWriter::setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const {
    CuboidShape imageShape = image.getShape();
    assert(imageShape.depth != 0|| imageShape.width != 0 || imageShape.height != 0 && "Cannot set subimage: Image3D has invalid dimensions");

    std::unique_lock<std::mutex> lock(writerMutex);

    int bufferIndex;
    bufferIndex = getStripIndex(coord);
    try{
        if (bufferIndex != -1){
            copyToTile(image, coord, bufferIndex);
        }
        else{
            createNewTile(coord);

            bufferIndex = getStripIndex(coord);
            copyToTile(image, coord, bufferIndex);
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

void TiffWriter::createNewTile(const BoxCoordWithPadding& coord) const {

    ImageBuffer tile;
    BoxCoordWithPadding source{
        BoxCoord{
            CuboidPosition{0,0,coord.box.position.depth},
            CuboidShape{imageShape.width, imageShape.height, coord.box.dimensions.depth}},
        coord.padding
    };
    tile.source = source;

    Image3D image(tile.source.box.dimensions, -1.0f);
    tile.image = std::move(image);

    tileBuffer.push_back(std::move(tile));

}



int TiffWriter::getStripIndex(const BoxCoordWithPadding& coord) const {
    BoxCoordWithPadding actualCoord = coord;
    actualCoord.padding.before = CuboidShape(0,0,0);
    actualCoord.padding.after = CuboidShape(0,0,0); // padding doesnt matter here
    for (size_t i = 0; i < tileBuffer.size(); i++){
        if (coord.isWithin(tileBuffer.find(i).source)){
            return static_cast<int>(i);
        }
    }
    return -1;
}



void TiffWriter::copyToTile(const Image3D& image, const BoxCoordWithPadding& coord, int index) const {
    ImageBuffer& tile = tileBuffer.find(index);
    BoxCoord srcBox = coord.box;
    srcBox.position = srcBox.position - tile.source.box.position;
    BoxCoord cubeBox = BoxCoord{coord.padding.before, coord.box.dimensions};

    ImageOperations::insertCubeInImage(image, cubeBox, tile.image, srcBox);
    if (isTileFull(tile)){
        // writeTile(metaData.filename, tile);
        // tileBuffer.deleteIndex(index);

        // Add tile to ready queue instead of writing immediately
        readyToWriteQueue.push(index);

        // Process queue to write tiles in correct order
        processReadyToWriteQueue();
    }

}

bool TiffWriter::isTileFull(const ImageBuffer& strip) const {
    using MinMaxFilterType = itk::MinimumMaximumImageFilter<ImageType>;

    auto minMax = MinMaxFilterType::New();
    minMax->SetInput(strip.image.getItkImage());
    minMax->Update();
    float min = minMax->GetMinimum();
    return (min != -1.0f); // make sure that initialization is the smallest possible value (we dont expect negative values in real image)

}


// so that tiles that are further back in the image are not written before the first ones, which could happen as its async
void TiffWriter::processReadyToWriteQueue() const {
    // Process tiles in order - only write tiles that are next in sequence
    while (!readyToWriteQueue.empty()) {
        // Find the tile with the smallest z-position among ready tiles
        int nextTileIndex = -1;
        int minZPosition = INT_MAX;

        // Create a temporary queue to check all ready tiles
        std::queue<int> tempQueue = readyToWriteQueue;
        std::vector<int> queuedTiles;

        while (!tempQueue.empty()) {
            int tileIndex = tempQueue.front();
            tempQueue.pop();
            queuedTiles.push_back(tileIndex);

            const ImageBuffer& tile = tileBuffer.find(tileIndex);
            int64_t zPos = tile.source.box.position.depth;

            // Check if this tile is the next one to write
            if (zPos <= static_cast<int64_t>(writtenToDepth) && zPos < minZPosition) {
                minZPosition = zPos;
                nextTileIndex = tileIndex;
            }
        }

        // If we found a tile that can be written next, write it
        if (nextTileIndex != -1) {
            const ImageBuffer& tileToWrite = tileBuffer.find(nextTileIndex);
            writeTile(outputFilename, tileToWrite);

            // Remove the written tile from buffer and queue
            tileBuffer.deleteIndex(nextTileIndex);

            // Rebuild the queue without the written tile
            std::queue<int> newQueue;
            for (int idx : queuedTiles) {
                if (idx != nextTileIndex) {
                    // Adjust indices since we deleted an element
                    int adjustedIdx = idx;
                    if (idx > nextTileIndex) {
                        adjustedIdx--;
                    }
                    newQueue.push(adjustedIdx);
                }
            }
            readyToWriteQueue = newQueue;
        } else {
            // No tile can be written yet, break the loop
            break;
        }
    }

}

bool TiffWriter::writeTile(const std::string& filename, const ImageBuffer& tile) const {
    int64_t y = tile.source.box.position.height;
    int64_t z = tile.source.box.position.depth;
    size_t height = tile.source.box.dimensions.height;
    size_t depth = tile.source.box.dimensions.depth;
    return writeToFile_(filename, static_cast<size_t>(z), depth, tile.image);

}

bool TiffWriter::writeToFile_(const std::string& filename, size_t z, size_t depth, const Image3D& layers) const {
    if (!tif) {
        throw TiffWriteException("TIFF file handle is null");
    }

    ImageMetaData metaData = extractMetaData(layers);

    for (size_t i = writtenToDepth - z; i < depth; ++i) {
        writeSliceToTiff(tif, layers, i, metaData, compressionConfig);
        if (!TIFFWriteDirectory(tif)) {
            throw TiffWriteException("Failed to set directory for slice " + std::to_string(i));
        }
    }

    writtenToDepth = z + depth;
    spdlog::info("Successfully saved ImageFileDirectory ({}): {} - {}", filename, z, z + depth);
    return true;
}


void TiffWriter::writeSliceToTiff(TIFF* tif, const Image3D& image, size_t sliceIndex, const ImageMetaData& metaData, const TiffCompressionConfig& compression){
    CuboidShape imgShape = image.getShape();
    if (sliceIndex >= imgShape.depth) {
        throw TiffWriteException("Slice index out of bounds: " + std::to_string(sliceIndex));
    }

    std::vector<float> sliceData;
    extractSliceData(image, sliceIndex, sliceData);

    if (sliceData.empty()) {
        throw TiffWriteException("Slice data is empty for slice index: " + std::to_string(sliceIndex));
    }

    setTiffFields(tif, metaData, compression);

    tmsize_t stripSize = TIFFStripSize(tif);
    tsize_t scanlineSize = TIFFScanlineSize(tif);

    if (stripSize <= 0) {
        throw TiffWriteException("Invalid strip size: " + std::to_string(stripSize));
    }

    char* buf = static_cast<char*>(_TIFFmalloc(stripSize));
    if (!buf) {
        throw TiffMemoryException("Memory allocation failed for strip buffer");
    }

    tmsize_t totalBytes = static_cast<tmsize_t>(scanlineSize) * static_cast<tmsize_t>(imgShape.height);

    if (totalBytes <= stripSize) {
        tmsize_t bytesToWrite = static_cast<tmsize_t>(scanlineSize) * static_cast<tmsize_t>(imgShape.height);
        memcpy(buf, sliceData.data(), bytesToWrite);
        if (TIFFWriteEncodedStrip(tif, 0, buf, bytesToWrite) == -1) {
            _TIFFfree(buf);
            throw TiffWriteException("Failed to write encoded strip for slice " + std::to_string(sliceIndex));
        }
    } else {
        uint32_t rowsPerStrip = 0;
        TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
        if (rowsPerStrip == 0) rowsPerStrip = static_cast<uint32_t>(imgShape.height);

        uint32_t stripCount = (imgShape.height + rowsPerStrip - 1) / rowsPerStrip;
        tmsize_t offset = 0;

        for (uint32_t s = 0; s < stripCount; ++s) {
            uint32_t rowsInThisStrip = std::min(rowsPerStrip, static_cast<uint32_t>(imgShape.height) - s * rowsPerStrip);
            tmsize_t bytesInStrip = static_cast<tmsize_t>(scanlineSize) * rowsInThisStrip;

            memcpy(buf, sliceData.data() + offset / sizeof(float), bytesInStrip);
            if (TIFFWriteEncodedStrip(tif, s, buf, bytesInStrip) == -1) {
                _TIFFfree(buf);
                throw TiffWriteException("Failed to write encoded strip " + std::to_string(s) + " for slice " + std::to_string(sliceIndex));
            }
            offset += bytesInStrip;
        }
    }

    _TIFFfree(buf);
}

void TiffWriter::setTiffFields(TIFF* tif, const ImageMetaData& metaData, const TiffCompressionConfig& compression){
    try {
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(metaData.imageWidth));
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(metaData.imageLength));
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, metaData.bitsPerSample);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, metaData.samplesPerPixel);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, metaData.planarConfig);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, metaData.photometricInterpretation);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, metaData.sampleFormat);

        uint32_t rowsPerStrip = static_cast<uint32_t>(metaData.imageLength);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

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

    // Make a copy of ap for size estimation since vsnprintf may consume it
    va_list ap_copy;
    va_copy(ap_copy, ap);
    int required = vsnprintf(nullptr, 0, fmt, ap_copy);
    va_end(ap_copy);

    if (required < 0) {
        // Fallback: log the raw format string if formatting failed
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
    uint64_t imageBytes = size.getVolume() * 4; // pixels are float

    // Create or open the TIFF file
    const uint64_t TWO_GIGABYTES = 2ULL * 1024 * 1024 * 1024;

    const char* mode = (imageBytes >= TWO_GIGABYTES) ? "w8" : "w";
    TIFF* tif = TIFFOpen(filename, mode);
    if (!tif) {
        throw TiffFileOpenException(filename);
    }
    return tif;
}

bool TiffWriter::writeToFile(const std::string& filename, const Image3D& image, TiffCompressionConfig config) {
    try {
        ImageMetaData metadata = extractMetaData(image);
        CuboidShape imgShape = image.getShape();

        if (imgShape.depth == 0 || imgShape.width == 0 || imgShape.height == 0) {
            throw TiffWriteException("Cannot write Image3D: Invalid image dimensions");
        }

        TIFFSetWarningHandler(TiffWriter::customTifWarningHandler);
        TIFF* tif = openTiff(filename.c_str(), imgShape);

        spdlog::debug("Writing TIFF: compression={}, level={}",
            TiffCompressionConfig::compressionToString(config.compressionScheme), config.compressionLevel);

        ImageMetaData metaData = extractMetaData(image);

        for (size_t z_index = 0; z_index < imgShape.depth; ++z_index) {
            writeSliceToTiff(tif, image, z_index, metaData, config);
            if (!TIFFWriteDirectory(tif)) {
                throw TiffWriteException("Failed to set directory for slice " + std::to_string(z_index));
            }
        }
        //
        // for (size_t zIndex = 0; zIndex < imgShape.depth; ++zIndex) {
        //     std::vector<float> sliceData;
        //     extractSliceData(image, zIndex, sliceData);
        //
        //     TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(metadata.imageWidth));
        //     TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(metadata.imageLength));
        //     TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, metadata.bitsPerSample);
        //     TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, metadata.samplesPerPixel);
        //     TIFFSetField(tif, TIFFTAG_PLANARCONFIG, metadata.planarConfig);
        //     TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, metadata.photometricInterpretation);
        //     TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, metadata.sampleFormat);
        //
        //     uint32_t rowsPerStrip = static_cast<uint32_t>(metadata.imageLength);
        //     TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);
        //
        //     TIFFSetField(tif, TIFFTAG_COMPRESSION, config.compressionScheme);
        //
        //     if (config.compressionScheme == COMPRESSION_LZW || config.compressionScheme == COMPRESSION_DEFLATE) {
        //         if (metadata.sampleFormat == SAMPLEFORMAT_IEEEFP) {
        //             TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
        //         } else {
        //             TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
        //         }
        //     }
        //
        //     if (config.compressionScheme == COMPRESSION_DEFLATE && config.compressionLevel >= 0) {
        //         TIFFSetField(tif, TIFFTAG_ZIPQUALITY, config.compressionLevel);
        //     }
        //
        //     tmsize_t stripSize = TIFFStripSize(tif);
        //     tsize_t scanlineSize = TIFFScanlineSize(tif);
        //
        //     char* buf = static_cast<char*>(_TIFFmalloc(stripSize));
        //     if (!buf) {
        //         TIFFClose(tif);
        //         throw TiffMemoryException("Memory allocation failed for strip buffer");
        //     }
        //
        //     tmsize_t bytesToWrite = static_cast<tmsize_t>(scanlineSize) * static_cast<tmsize_t>(imgShape.height);
        //     memcpy(buf, sliceData.data(), bytesToWrite);
        //
        //     if (TIFFWriteEncodedStrip(tif, 0, buf, bytesToWrite) == -1) {
        //         _TIFFfree(buf);
        //         TIFFClose(tif);
        //         throw TiffWriteException("Failed to write encoded strip for slice " + std::to_string(zIndex));
        //     }
        //
        //     _TIFFfree(buf);
        //
        //     if (!TIFFWriteDirectory(tif)) {
        //         TIFFClose(tif);
        //         throw TiffWriteException("Failed to set directory for slice " + std::to_string(zIndex));
        //     }
        // }

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
        // This function returns the ITK pixel type identifier
        // For simplicity, we'll use the metadata directly since ITK uses templated types
        if (metadata.bitsPerSample == 8) {
            return 8;  // 8-bit unsigned
        } else if (metadata.bitsPerSample == 16) {
            return 16; // 16-bit unsigned
        } else if (metadata.bitsPerSample == 32) {
            return 32; // 32-bit float
        } else {
            throw TiffMetadataException("Unsupported bit depth from metadata: " + std::to_string(metadata.bitsPerSample));
        }

}

void TiffWriter::extractSliceData(const Image3D& image, size_t sliceIndex, std::vector<float>& sliceData) {
    CuboidShape shape = image.getShape();
    if (sliceIndex >= shape.depth) {
        throw TiffWriteException("Slice index out of bounds: " + std::to_string(sliceIndex));
    }

    sliceData.resize(shape.width * shape.height);

    // Use ITK slice iterator to extract slice data
    ImageType::Pointer itkImage = image.getItkImage();

    // Define the slice region
    ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = static_cast<itk::IndexValueType>(sliceIndex);

    ImageType::SizeType size;
    size[0] = shape.width;
    size[1] = shape.height;
    size[2] = 1;

    ImageType::RegionType sliceRegion;
    sliceRegion.SetIndex(start);
    sliceRegion.SetSize(size);

    itk::ImageRegionIterator<ImageType> it(itkImage, sliceRegion);

    size_t pixelIndex = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++pixelIndex) {
        sliceData[pixelIndex] = it.Get();
    }

}

void TiffWriter::convertSliceDataToTargetType(const std::vector<float>& sourceData,
                                            std::vector<uint8_t>& targetData,
                                            size_t width, size_t height,
                                            const ImageMetaData& metadata) {
    size_t numPixels = width * height;

    if (metadata.bitsPerSample == 8) {
        // Convert float to 8-bit unsigned
        targetData.resize(numPixels);
        for (size_t i = 0; i < numPixels; ++i) {
            // Assume source data is in [0,1] range, scale to [0,255]
            float scaledValue = sourceData[i] * 255.0f;
            targetData[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, scaledValue)));
        }
    } else if (metadata.bitsPerSample == 16) {
        // Convert float to 16-bit unsigned
        targetData.resize(numPixels * 2);
        uint16_t* data16 = reinterpret_cast<uint16_t*>(targetData.data());
        for (size_t i = 0; i < numPixels; ++i) {
            // Assume source data is in [0,1] range, scale to [0,65535]
            float scaledValue = sourceData[i] * 65535.0f;
            data16[i] = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, scaledValue)));
        }
    } else if (metadata.bitsPerSample == 32) {
        // Keep as 32-bit float
        targetData.resize(numPixels * 4);
        float* data32 = reinterpret_cast<float*>(targetData.data());
        for (size_t i = 0; i < numPixels; ++i) {
            data32[i] = sourceData[i];
        }
    } else {
        throw TiffMetadataException("Unsupported bit depth: " + std::to_string(metadata.bitsPerSample));
    }

}
