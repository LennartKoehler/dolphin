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

#include <opencv2/imgproc.hpp>
#include "IO/TiffWriter.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <cstdarg>
#include <climits>
#include <queue>
#include "deconvolution/Preprocessor.h"
#include "UtlImage.h"
#include "deconvolution/Postprocessor.h"


namespace fs = std::filesystem;

// Constructor
TiffWriter::TiffWriter(const std::string& filename, const ImageMetaData& metadata) {
    this->metaData = metadata;
    metaData.filename = filename;
    
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

const ImageMetaData& TiffWriter::getMetaData() const {
    return metaData;
}

bool TiffWriter::setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const {
    if (image.slices.empty()) {
        std::cerr << "[ERROR] Cannot set subimage: Image3D has no slices" << std::endl;
        return false;
    }
    
    // Validate the image dimensions
    if (image.slices[0].cols <= 0 || image.slices[0].rows <= 0) {
        std::cerr << "[ERROR] Cannot set subimage: Invalid image dimensions" << std::endl;
        return false;
    }
    
    std::unique_lock<std::mutex> lock(writerMutex); //TESTVALUE
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
            RectangleShape{metaData.imageWidth, metaData.imageLength, coord.box.dimensions.depth}},
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

void TiffWriter::postprocessChannel(Image3D& image){
    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    for (const auto& slice : image.slices) {
        cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO);
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);
    }
    float epsilon = 1e-6; //TESTVALUE
    for (auto& slice : image.slices) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));
        cv::threshold(slice, slice, epsilon, 0.0, cv::THRESH_TOZERO);
    }
}

bool TiffWriter::copyToTile(const Image3D& image, const BoxCoordWithPadding& coord, int index) const {
    ImageBuffer& tile = tileBuffer.find(index);
    BoxCoord srcBox = coord.box;
    srcBox.position = srcBox.position - tile.source.box.position;
    BoxCoord cubeBox = BoxCoord{coord.padding.before, coord.box.dimensions};

    Postprocessor::insertCubeInImage(image, cubeBox, tile.image, srcBox);
    
    tile.interactedValue += coord.box.dimensions.width * coord.box.dimensions.height;
    
    if (isTileFull(tile)){
        writeTile(metaData.filename, tile);
        tileBuffer.deleteIndex(index);

        // // Add tile to ready queue instead of writing immediately
        // readyToWriteQueue.push(index);
        
        // // Process queue to write tiles in correct order
        // processReadyToWriteQueue();
    }
    return true;
}

bool TiffWriter::isTileFull(const ImageBuffer& strip) const {
    cv::Mat mask = (strip.image.slices[0] == -1.0f); // since im writing tiles, the depth dimension shouldnt matter and i can take first slice
    int remaining = cv::countNonZero(mask);
    return remaining == 0;
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
            writeTile(metaData.filename, tileToWrite);
            
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
    return saveToFile(filename, y, z, height, depth, tile.image);
}

bool TiffWriter::saveToFile(const std::string& filename, int y, int z, int height, int depth, const Image3D& layers) const {



    // TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, stripHeight);
    if (!tif) {
        std::cerr << "[ERROR] Cannot create TIFF file: " << filename << std::endl;
        return false;
    }
    
     
    // Write each slice as a separate directory in the TIFF file

    for (size_t i = writtenToDepth - z; i < depth; ++i) {

        // Write the slice
        if (!writeImageToTiff(layers.slices[i], i + z, y)) {
            TIFFClose(tif);
            return false;
        }
         // Set directory for this slice
        if (!TIFFWriteDirectory(tif)) { //TESTVALUE TIFFSetDirectory
            TIFFClose(tif);
            std::cerr << "[ERROR] Failed to set directory for slice " << i << std::endl;
            return false;
        }
        

    }
    // directories.clear();
    // TIFFClose(tif);
    writtenToDepth = z + depth;

    std::cout << "[INFO] Successfully saved TIFF file: " << filename << std::endl;
    return true;
}


bool TiffWriter::writeImageToTiff(const cv::Mat& image, int directoryIndex, int yOffset) const {
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot write empty image to TIFF" << std::endl;
        return false;
    }
    
    // Only set TIFF fields for the first directory (directoryIndex == 0) or when starting a new directory

    
    // if (directories.find(directoryIndex) == directories.end()) {
        // directories[directoryIndex] = true;
        
    // Determine TIFF data type based on OpenCV type
    uint16_t bitsPerSample = 8;
    uint16_t samplesPerPixel = 1;
    int cvType = CV_32F; //TESTVALUE image.type();
    
    switch (CV_MAT_DEPTH(cvType)) {
        case CV_8U:
            bitsPerSample = 8;
            break;
        case CV_16U:
            bitsPerSample = 16;
            break;
        case CV_32F:
            bitsPerSample = 32;
            break;
        default:
            std::cerr << "[ERROR] Unsupported OpenCV data type: " << cvType << std::endl;
        }
    int sampleFormat;
    switch (bitsPerSample) {
        case 8: sampleFormat = SAMPLEFORMAT_UINT; break;
        case 16: sampleFormat = SAMPLEFORMAT_UINT; break;
        case 32: sampleFormat = SAMPLEFORMAT_IEEEFP; break;
        default: sampleFormat = SAMPLEFORMAT_UINT; break;
    }
    samplesPerPixel = CV_MAT_CN(cvType);
    
    // Process description to remove min/max/mode lines
    std::istringstream iss(this->metaData.description);
    std::ostringstream oss;
    std::string line;
    while (std::getline(iss, line)) {
        // Prüfen, ob die Zeile "min=" oder "max=" enthält
        if (line.find("min=") == std::string::npos && line.find("max=") == std::string::npos && line.find("mode=") == std::string::npos) {
            oss << line << "\n";
        }
    }
    std::string cutted_description = oss.str();
    
    // Set TIFF fields only for new directories
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image.cols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.rows);
    // TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 50); //TESTVALUE

    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);

    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, this->metaData.planarConfig);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, this->metaData.photometricInterpretation);
    if(!(this->metaData.description == "" || this->metaData.description.empty())){
        TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, cutted_description.c_str());
    }
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sampleFormat);
    

    tsize_t scanlineSize = TIFFScanlineSize(tif);
    
    // Allocate buffer for scanlines
    char* buf = (char*)_TIFFmalloc(scanlineSize);
    if (!buf) {
        std::cerr << "[ERROR] Memory allocation failed for scanline buffer" << std::endl;
        return false;
    }

    // if(!(this->metaData.description == "" || this->metaData.description.empty())){
    //     TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, cutted_description.c_str());
    // }
    // Calculate scanline size
    // // Convert image to appropriate format for writing
    // cv::Mat imageToWrite;
    // if (image.type() != CV_8UC(samplesPerPixel) && 
    //     image.type() != CV_16UC(samplesPerPixel) && 
    //     image.type() != CV_32FC(samplesPerPixel)) {
    //     // Convert to 8-bit unsigned for compatibility
    //     double minVal, maxVal;
    //     cv::minMaxLoc(image, &minVal, &maxVal);
    //     if (maxVal > minVal) {
    //         image.convertTo(imageToWrite, CV_8UC(samplesPerPixel), 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));
    //     } else {
    //         imageToWrite = cv::Mat::zeros(height, width, CV_8UC(samplesPerPixel));
    //     }
    // } else {
    //     imageToWrite = image.clone();
    // }
    
    // Write scanlines
    for (uint32_t row = 0; row < image.rows; ++row) {
        memcpy(buf, image.ptr(row), scanlineSize);
        if (TIFFWriteScanline(tif, buf, row, 0) == -1) { //TESTVALUE + offset
            _TIFFfree(buf);
            std::cerr << "[ERROR] Failed to write scanline " << row << " in directory " << directoryIndex << std::endl;
            return false;
        }
    }
    
    // Clean up
    _TIFFfree(buf);
    
    return true;
}

void TiffWriter::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    // Suppress TIFF warnings
    // printf(fmt, ap); // Uncomment to show warnings
}



// Static method for writing entire Image3D to file
bool TiffWriter::writeToFile(const std::string& filename, const Image3D& image, const ImageMetaData& metadata) {
    if (image.slices.empty()) {
        std::cerr << "[ERROR] Cannot write Image3D: No slices available" << std::endl;
        return false;
    }
    
    // Validate image dimensions
    if (image.slices[0].cols <= 0 || image.slices[0].rows <= 0) {
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
    
    // Determine OpenCV data type from first slice
    int cvType = image.slices[0].type();
    uint16_t bitsPerSample = 8;
    uint16_t samplesPerPixel = 1;
    
    switch (CV_MAT_DEPTH(cvType)) {
        case CV_8U:
            bitsPerSample = 8;
            break;
        case CV_16U:
            bitsPerSample = 16;
            break;
        case CV_32F:
            bitsPerSample = 32;
            break;
        default:
            std::cerr << "[ERROR] Unsupported OpenCV data type: " << cvType << std::endl;
            TIFFClose(tif);
            return false;
    }
    
    int sampleFormat;
    switch (bitsPerSample) {
        case 8: sampleFormat = SAMPLEFORMAT_UINT; break;
        case 16: sampleFormat = SAMPLEFORMAT_UINT; break;
        case 32: sampleFormat = SAMPLEFORMAT_IEEEFP; break;
        default: sampleFormat = SAMPLEFORMAT_UINT; break;
    }
    samplesPerPixel = CV_MAT_CN(cvType);
    
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
    for (size_t zIndex = 0; zIndex < image.slices.size(); ++zIndex) {
        const cv::Mat& slice = image.slices[zIndex];
        
        // Set TIFF fields for this directory
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, slice.cols);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, slice.rows);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
        TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, metadata.planarConfig);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, metadata.photometricInterpretation);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sampleFormat);
        
        if (!cutted_description.empty()) {
            TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, cutted_description.c_str());
        }
        
        // Set resolution if available
        if (metadata.xResolution > 0 && metadata.yResolution > 0) {
            TIFFSetField(tif, TIFFTAG_XRESOLUTION, metadata.xResolution);
            TIFFSetField(tif, TIFFTAG_YRESOLUTION, metadata.yResolution);
            TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, metadata.resolutionUnit);
        }
        
        tsize_t scanlineSize = TIFFScanlineSize(tif);
        
        // Allocate buffer for scanlines
        char* buf = (char*)_TIFFmalloc(scanlineSize);
        if (!buf) {
            std::cerr << "[ERROR] Memory allocation failed for scanline buffer" << std::endl;
            TIFFClose(tif);
            return false;
        }
        
        // Write scanlines
        for (uint32_t row = 0; row < slice.rows; ++row) {
            memcpy(buf, slice.ptr(row), scanlineSize);
            if (TIFFWriteScanline(tif, buf, row, 0) == -1) {
                _TIFFfree(buf);
                TIFFClose(tif);
                std::cerr << "[ERROR] Failed to write scanline " << row << " in slice " << zIndex << std::endl;
                return false;
            }
        }
        
        // Clean up buffer
        _TIFFfree(buf);
        
        // Set directory for next slice (except for the last one)
        if (zIndex < image.slices.size() - 1) {
            if (!TIFFWriteDirectory(tif)) {
                TIFFClose(tif);
                std::cerr << "[ERROR] Failed to set directory for slice " << zIndex << std::endl;
                return false;
            }
        }
    }
    
    // Close the TIFF file
    TIFFClose(tif);
    
    std::cout << "[INFO] Successfully wrote Image3D to TIFF file: " << filename << std::endl;
    return true;
}



// void TiffWriter::mergeSubimages() {
//     if (strips.empty()) return;
    
//     // Create a merged image3D to hold all the data
//     Image3D mergedImage;
//     mergedImage.slices.resize(strips[0].slices.size());
    
//     // Initialize merged slices with zeros
//     for (size_t i = 0; i < mergedImage.slices.size(); ++i) {
//         mergedImage.slices[i] = cv::Mat::zeros(metaData.imageLength, metaData.imageWidth, CV_32FC(strips[0].slices[0].channels()));
//     }
    
//     // Copy each subimage into the merged image at the correct position
//     for (size_t i = 0; i < strips.size(); ++i) {
//         const Image3D& subimage = strips[i];
//         const BoxCoordWithPadding& coord = subimageCoords[i];
//         const cv::Mat& mask = masks[i];
        
//         // Apply mask to subimage if mask is provided
//         cv::Mat processedSubimage = subimage;
//         if (!mask.empty()) {
//             processedSubimage = applyMaskToImage(subimage.slices[0], mask);
//         }
        
//         // Copy each slice of the subimage to the merged image
//         for (size_t z = 0; z < subimage.slices.size(); ++z) {
//             int zPos = coord.box.position.depth + z;
//             if (zPos >= 0 && zPos < static_cast<int>(mergedImage.slices.size())) {
//                 // Define the ROI in the merged image
//                 cv::Rect roi(coord.box.position.width, coord.box.position.height, 
//                            subimage.slices[z].cols, subimage.slices[z].rows);
                
//                 // Copy the subimage slice to the merged image
//                 if (roi.x >= 0 && roi.y >= 0 && 
//                     roi.x + roi.width <= metaData.imageWidth && 
//                     roi.y + roi.height <= metaData.imageLength) {
//                     subimage.slices[z].copyTo(mergedImage.slices[zPos](roi));
//                 }
//             }
//         }
//     }
    
//     // Replace the first subimage with the merged image
//     if (!strips.empty()) {
//         strips[0] = mergedImage;
//         subimageCoords.resize(1);
//         masks.resize(1);
//     }
// }

