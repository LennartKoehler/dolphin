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

#include "IO/TiffWriter.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>

#include <filesystem>
#include <fstream>
#include <cstdarg>
#include <climits>
#include <queue>
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkMinimumMaximumImageFilter.h"

namespace fs = std::filesystem;

// Constructor
TiffWriter::TiffWriter(const std::string& filename, const RectangleShape& imageShape) 
    : outputFilename(filename),
    imageShape(imageShape){
    // this->metaData = metadata;
    // metaData.filename = filename;
    
    TIFFSetWarningHandler(TiffWriter::customTifWarningHandler);
    
    // Create or open the TIFF file
    TIFF* tif = TIFFOpen(filename.c_str(), "w");
    this->tif = tif;
    


}

// Destructor
TiffWriter::~TiffWriter() {   
    assert (tileBuffer.size() == 0 && "TileBuffer not empty but done writing, should not happen"); 
    TIFFClose(tif);
}

// const ImageMetaData& TiffWriter::getMetaData() const {
//     return metaData;
// }

bool TiffWriter::setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const {
    RectangleShape imageShape = image.getShape();
    if (imageShape.depth == 0 || imageShape.width == 0 || imageShape.height == 0) {
        std::cerr << "[ERROR] Cannot set subimage: Image3D has invalid dimensions" << std::endl;
        return false;
    }
    
    std::unique_lock<std::mutex> lock(writerMutex);
    

    int bufferIndex;
    bufferIndex = getStripIndex(coord);
    if (bufferIndex != -1){
        return copyToTile(image, coord, bufferIndex);
    }
    createNewTile(coord);

    bufferIndex = getStripIndex(coord);
    copyToTile(image, coord, bufferIndex);
    return true;
}

void TiffWriter::createNewTile(const BoxCoordWithPadding& coord) const {
    ImageBuffer tile;
    BoxCoordWithPadding source{
        BoxCoord{
            RectangleShape{0,0,coord.box.position.depth},
            RectangleShape{imageShape.width, imageShape.height, coord.box.dimensions.depth}},
        coord.padding
    };
    tile.source = source;

    Image3D image(tile.source.box.dimensions);
    tile.image = std::move(image);

    tileBuffer.push_back(std::move(tile));
}



int TiffWriter::getStripIndex(const BoxCoordWithPadding& coord) const {
    BoxCoordWithPadding actualCoord = coord;
    actualCoord.padding.before = RectangleShape(0,0,0);
    actualCoord.padding.after = RectangleShape(0,0,0); // padding doesnt matter here
    for (int i = 0; i < tileBuffer.size(); i++){
        if (coord.isWithin(tileBuffer.find(i).source)){
            return i;
        }
    }
    return -1;
}



bool TiffWriter::copyToTile(const Image3D& image, const BoxCoordWithPadding& coord, int index) const {
    ImageBuffer& tile = tileBuffer.find(index);
    BoxCoord srcBox = coord.box;
    srcBox.position = srcBox.position - tile.source.box.position;
    BoxCoord cubeBox = BoxCoord{coord.padding.before, coord.box.dimensions};

    Postprocessor::insertCubeInImage(image, cubeBox, tile.image, srcBox);
    if (isTileFull(tile)){
        // writeTile(metaData.filename, tile);
        // tileBuffer.deleteIndex(index);

        // Add tile to ready queue instead of writing immediately
        readyToWriteQueue.push(index);
        
        // Process queue to write tiles in correct order
        processReadyToWriteQueue();
    }
    return true;
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
            int zPos = tile.source.box.position.depth;
            
            // Check if this tile is the next one to write
            if (zPos <= writtenToDepth && zPos < minZPosition) {
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
    int y = tile.source.box.position.height;
    int z = tile.source.box.position.depth;
    int height = tile.source.box.dimensions.height;
    int depth = tile.source.box.dimensions.depth;
    return saveToFile(filename, z, depth, tile.image);
}

bool TiffWriter::saveToFile(const std::string& filename, int z, int depth, const Image3D& layers) const {



    // TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, stripHeight);
    if (!tif) {
        std::cerr << "[ERROR] Cannot create TIFF file: " << filename << std::endl;
        return false;
    }
    
     
    // Write each slice as a separate directory in the TIFF file

    for (size_t i = writtenToDepth - z; i < depth; ++i) {

        // Write the slice
        if (!writeSliceToTiff(tif, layers, i)) {
            TIFFClose(tif);
            return false;
        }
         // Set directory for this slice
        if (!TIFFWriteDirectory(tif)) {
            TIFFClose(tif);
            std::cerr << "[ERROR] Failed to set directory for slice " << i << std::endl;
            return false;
        }
        

    }
    // directories.clear();
    // TIFFClose(tif);
    writtenToDepth = z + depth;

    std::cout << "[INFO] Successfully saved ImageFileDirectory (" << filename << "): " << z << " - " << z + depth << std::endl;
    return true;
}


bool TiffWriter::writeSliceToTiff(TIFF* tif, const Image3D& image,  int sliceIndex){
    RectangleShape imageShape = image.getShape();
    if (sliceIndex >= imageShape.depth) {
        std::cerr << "[ERROR] Slice index out of bounds: " << sliceIndex << std::endl;
        return false;
    }
    
    // Extract slice data
    std::vector<float> sliceData;
    extractSliceData(image, sliceIndex, sliceData);
    
    if (sliceData.empty()) {
        std::cerr << "[ERROR] Cannot write empty slice to TIFF" << std::endl;
        return false;
    }
    
    ImageMetaData metaData = extractMetaData(image);
    setTiffFields(tif, metaData); 

    
    tsize_t scanlineSize = TIFFScanlineSize(tif);
    
    // Allocate buffer for scanlines
    char* buf = (char*)_TIFFmalloc(scanlineSize);
    if (!buf) {
        std::cerr << "[ERROR] Memory allocation failed for scanline buffer" << std::endl;
        return false;
    }
    
    // Write scanlines
    // size_t bytesPerPixel = bitsPerSample / 8 * samplesPerPixel;
    for (uint32_t row = 0; row < imageShape.height; ++row) {
        size_t srcOffset = row * imageShape.width ;
        memcpy(buf, &sliceData[srcOffset], scanlineSize);
        if (TIFFWriteScanline(tif, buf, row, 0) == -1) {
            _TIFFfree(buf);
            std::cerr << "[ERROR] Failed to write scanline " << row << std::endl;
            return false;
        }
    }
    
    // Clean up
    _TIFFfree(buf);
    
    return true;
}

void TiffWriter::setTiffFields(TIFF* tif, const ImageMetaData& metaData){
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, metaData.imageWidth);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, metaData.imageLength);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, metaData.bitsPerSample);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, metaData.samplesPerPixel);
    // TIFFSetField(tif, TIFFTAG_ORIENTATION, metaData.orientation);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, metaData.planarConfig);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, metaData.photometricInterpretation);
    // if(!(metaData.description == "" || metaData.description.empty())){
    //     TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, cutted_description.c_str());
    // }
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, metaData.sampleFormat);
}

void TiffWriter::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    // Suppress TIFF warnings
    // printf(fmt, ap); // Uncomment to show warnings
}


ImageMetaData TiffWriter::extractMetaData(const Image3D& image){
    // Convert to target type based on metadata
    // std::vector<uint8_t> convertedData;
    // convertSliceDataToTargetType(sliceData, convertedData, imageShape.width, imageShape.height, metaData);
    
    // Use metadata values for TIFF fields
    // uint16_t bitsPerSample = metaData.bitsPerSample;
    // uint16_t samplesPerPixel = metaData.samplesPerPixel;
    // int sampleFormat = metaData.sampleFormat;
    
    // Process description to remove min/max/mode lines
    // std::istringstream iss(metaData.description);
    // std::ostringstream oss;
    // std::string line;
    // while (std::getline(iss, line)) {
    //     if (line.find("min=") == std::string::npos && line.find("max=") == std::string::npos && line.find("mode=") == std::string::npos) {
    //         oss << line << "\n";
    //     }
    // }
    // std::string cutted_description = oss.str();
    ImageMetaData metaData;
    RectangleShape size = image.getShape();
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
// Static method for writing entire Image3D to file
bool TiffWriter::writeToFile(const std::string& filename, const Image3D& image) {
    ImageMetaData metadata = extractMetaData(image);
    RectangleShape imageShape = image.getShape();
    
    if (imageShape.depth == 0 || imageShape.width == 0 || imageShape.height == 0) {
        std::cerr << "[ERROR] Cannot write Image3D: Invalid image dimensions" << std::endl;
        return false;
    }
    
    // Set up warning handler
    TIFFSetWarningHandler(TiffWriter::customTifWarningHandler);
    
    // Create or open the TIFF file
    TIFF* tif = TIFFOpen(filename.c_str(), "w");
    if (!tif) {
        std::cerr << "[ERROR] Cannot create TIFF file: " << filename << std::endl;
        return false;
    }
    
    // Use metadata values for TIFF fields
    uint16_t bitsPerSample = metadata.bitsPerSample;
    uint16_t samplesPerPixel = metadata.samplesPerPixel;
    int sampleFormat = metadata.sampleFormat;
    
    // Process description to remove min/max/mode lines
    std::string cutted_description = metadata.description;
    if (!metadata.description.empty()) {
        std::istringstream iss(metadata.description);
        std::ostringstream oss;
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find("min=") == std::string::npos &&
                line.find("max=") == std::string::npos &&
                line.find("mode=") == std::string::npos) {
                oss << line << "\n";
            }
        }
        cutted_description = oss.str();
    }
    

    // Write each slice as a separate directory in the TIFF file
    for (int zIndex = 0; zIndex < imageShape.depth - 1; ++zIndex) {
        TiffWriter::writeSliceToTiff(tif, image, zIndex);
        // Set directory for next slice (except for the last one)
        if (!TIFFWriteDirectory(tif)) {
            TIFFClose(tif);
            std::cerr << "[ERROR] Failed to set directory for slice " << zIndex << std::endl;
            return false;
        }
    }
    
    // Close the TIFF file
    TIFFClose(tif);
    
    std::cout << "[INFO] Successfully wrote Image3D to TIFF file: " << filename << std::endl;
    return true;
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
        std::cerr << "[ERROR] Unsupported bit depth from metadata: " << metadata.bitsPerSample << std::endl;
        return 8; // fallback to 8-bit
    }
}

void TiffWriter::extractSliceData(const Image3D& image, int sliceIndex, std::vector<float>& sliceData) {
    RectangleShape shape = image.getShape();
    if (sliceIndex >= shape.depth) {
        std::cerr << "[ERROR] Slice index out of bounds: " << sliceIndex << std::endl;
        return;
    }
    
    sliceData.resize(shape.width * shape.height);
    
    // Use ITK slice iterator to extract slice data
    ImageType::Pointer itkImage = image.getItkImage();
    
    // Define the slice region
    ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = sliceIndex;
    
    ImageType::SizeType size;
    size[0] = shape.width;
    size[1] = shape.height;
    size[2] = 1;
    
    ImageType::RegionType sliceRegion;
    sliceRegion.SetIndex(start);
    sliceRegion.SetSize(size);
    
    itk::ImageRegionIterator<ImageType> it(itkImage, sliceRegion);
    
    int pixelIndex = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++pixelIndex) {
        sliceData[pixelIndex] = it.Get();
    }
}

void TiffWriter::convertSliceDataToTargetType(const std::vector<float>& sourceData, 
                                            std::vector<uint8_t>& targetData, 
                                            int width, int height,
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
        std::cerr << "[ERROR] Unsupported bit depth: " << metadata.bitsPerSample << std::endl;
        // Fallback to 8-bit
        targetData.resize(numPixels);
        for (size_t i = 0; i < numPixels; ++i) {
            float scaledValue = sourceData[i] * 255.0f;
            targetData[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, scaledValue)));
        }
    }
}

