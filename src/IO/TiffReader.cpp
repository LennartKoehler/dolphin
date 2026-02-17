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

#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffExceptions.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdarg>
#include "dolphin/deconvolution/Preprocessor.h"
#include <chrono>
#include <thread>
#include <itkImageRegionIterator.h>
#include <spdlog/spdlog.h>
namespace fs = std::filesystem;



// Constructor with filename
TiffReader::TiffReader(const std::string& filename, int channel)
    : channel(channel)
{
    std::unique_lock<std::mutex> lock(mutex);
    // Set filename in metadata
    metaData.filename = filename;
    maxBufferMemory_bytes = 999999999; //TESTVALUE
    
    TIFFSetWarningHandler(customTifWarningHandler);
    
    try {
        metaData = readMetadata_(filename);

    } catch (const TiffException& e) {
        throw TiffFileOpenException(filename);
    }
    
    currentBufferMemory_bytes = 0;
    tif = TIFFOpen(filename.c_str(), "r");
    if (!tif) {
        throw TiffFileOpenException(filename);
    }
}

// Destructor
TiffReader::~TiffReader() {
    // Clean up resources if needed
    if (tif) {
        TIFFClose(tif);
        tif = nullptr;
    }
}


// Static method for reading entire TIFF file
std::optional<Image3D> TiffReader::readTiffFile(const std::string& filename, int channel) {
    try {
        TIFFSetWarningHandler(customTifWarningHandler);
        ImageMetaData metaData = readMetadata_(filename);
        
        Image3D image;
        BoxCoord fullImage{CuboidShape{0,0,0}, CuboidShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
        
        readSubimageFromTiffFileStatic(filename, metaData, fullImage.position.height, fullImage.position.depth,
                         fullImage.dimensions.height, fullImage.dimensions.depth, fullImage.dimensions.width, image, channel);

        
        // convertImageTo32F(image, metaData);
        return image;

    } catch (const TiffMemoryException& e) {
        spdlog::warn("Insufficient memory to read TIFF file {}: {}", filename, e.what());
        throw;
    } catch (const TiffException& e) {
        spdlog::error("{}", e.what());
        throw;
    } catch (const std::runtime_error& e) {
        spdlog::error("{}",e.what());
        throw;
    }
}

// Static method for extracting metadata


std::optional<ImageMetaData> TiffReader::readMetadata(const std::string& filename){
    try{
        return std::optional<ImageMetaData>(readMetadata_(filename));
    }
    catch(...){
        throw;
        return {};
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
            metaData.slices = metaData.totalImages + 1;
        }
        
        return metaData;
    } catch (...) {
        TIFFClose(tifFile);
        throw;
    }
}


void TiffReader::readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& image, int channel){
     
    TIFFSetWarningHandler(TiffReader::customTifWarningHandler);
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (!tif) {
        throw TiffFileOpenException(filename);
    }
    
    try {
        readSubimageFromTiffFile(tif, metaData, y, z, height, depth, width, image, channel);
        TIFFClose(tif);
        convertImageTo32F(image, metaData);
    } catch (const TiffException& e) {
        TIFFClose(tif);
        throw;
    } catch (const std::exception& e){
        TIFFClose(tif);
        throw TiffReadException(e.what());
    }
}




void TiffReader::readSubimageFromTiffFile(TIFF* tiffile, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& image, int channel) {
    try{
        if (!tiffile) {
            throw TiffException("TIFF File is not open");
        }
        
        // Validate channel parameter
        if (channel < 0) {
            spdlog::warn("Invalid channel {}, using channel 0", channel);
            channel = 0;
        }
        
        // there are different ways to store multiple channels, either by interleaved directories, or by multiple samples per pixels
        // i dotn think both are used together
        int ifdchannel = 0; // image file directory
        int sppchannel = 0; // samples per pixel
        if (metaData.linChannels > 1){
            ifdchannel = channel - 1; // channel is 1 based
            if (ifdchannel > metaData.linChannels - 1){
                throw TiffMetadataException("Specified channel " + std::to_string(channel) + 
                                        " larger than maximum number of image file directories: " + 
                                        std::to_string(metaData.linChannels));
            }
        }
        else if (metaData.samplesPerPixel > 1){
            sppchannel = channel - 1; // channel is 1 based
            if (sppchannel > metaData.samplesPerPixel - 1){
                throw TiffMetadataException("Specified channel " + std::to_string(channel) +
                                        " larger than maximum number of samples per pixel: " +
                                        std::to_string(metaData.samplesPerPixel));
            }
        }
        

        assert(y > 0 || z > 0 || height > 1 || depth > 1 && "Invalid regions coordinates to read from TiffFile");
        
        // Create ITK image with the specified dimensions
        CuboidShape imageShape(width, height, depth);
        image = Image3D(imageShape);
        
        // Read the specific region using scanline API
        tsize_t scanlineSize = TIFFScanlineSize(tiffile);
        char* buf = nullptr;
        try {
            buf = (char*)_TIFFmalloc(scanlineSize);
            if (!buf) {
                throw TiffMemoryException("Failed to allocate scanline buffer");
            }
        } catch (const TiffMemoryException&) {
            throw; // Re-throw specific memory exception
        } catch (...) {
            throw TiffMemoryException("Unexpected error during scanline buffer allocation");
        }
        
        // Create a temporary buffer for conversion
        std::vector<float> rowData(width);
        
        // Read each directory (z-slice) in the region
        for (uint32_t zIndex = z; zIndex < z + depth; zIndex++) {
            // Always set the directory for this z-slice (including z=0)

            int zIndexChannel = (zIndex * metaData.linChannels) + ifdchannel;
            if (!TIFFSetDirectory(tiffile, zIndexChannel)) {
                _TIFFfree(buf);
                throw TiffReadException("Failed to set directory for z-slice " + std::to_string(zIndex) + 
                                    " (channel " + std::to_string(zIndexChannel) + ")");
            }
            
            // Read only the required rows (scanlines)
            for (uint32_t yIndex = y; yIndex < y + height; yIndex++) {
                if (TIFFReadScanline(tiffile, buf, yIndex) == -1) {
                    _TIFFfree(buf);
                    throw TiffReadException("Failed to read scanline " + std::to_string(yIndex) +
                                        " in z-slice " + std::to_string(zIndex));
                }
                
                try {
                    convertScanlineToFloat(buf, rowData, width, metaData, sppchannel);
                } catch (const TiffException& e) {
                    _TIFFfree(buf);
                    throw TiffReadException("Error converting scanline at y=" + std::to_string(yIndex) +
                                        ", z=" + std::to_string(zIndex) + ": " + e.what());
                }
                
                // Set the row data into the ITK image
                image.setRow(yIndex - y, zIndex - z, rowData.data());

            }
        }
        
        _TIFFfree(buf);
        
        spdlog::info("Successfully read strip ({}): ({},{}) {}x{}", metaData.filename, y, z, height, depth);
    } catch (const TiffException& e) {
        throw e;
    } catch (const std::exception& e){
        throw TiffReadException(e.what());
    } catch (...){
        throw std::runtime_error("Unexpected error in tiffReader");
    }
}



void TiffReader::updateCurrentMemoryBuffer(size_t memory) const {
    // std::unique_lock<std::mutex> lock(mutex);
    currentBufferMemory_bytes = memory;
}

size_t TiffReader::getMemoryForShape(const CuboidShape& shape, const ImageMetaData& metaData){
    // Calculate memory requirement based on metaData's bit depth and samples per pixel
    size_t bytesPerPixel = (metaData.bitsPerSample / 8) * metaData.samplesPerPixel;
    return shape.getVolume() * bytesPerPixel;
}

Image3D TiffReader::managedReader(const BoxCoord& coord) const {
    // std::unique_lock<std::mutex> lock(mutex);
    size_t memorySize = getMemoryForShape(coord.dimensions, metaData);
    // memoryWaiter.wait(lock, [this, memorySize]() {
    //     return currentBufferMemory_bytes + memorySize < maxBufferMemory_bytes;
    // });//TESTVALUE
    Image3D result;
    readSubimageFromTiffFile(tif, metaData, coord.position.height, coord.position.depth, 
                coord.dimensions.height, coord.dimensions.depth, coord.dimensions.width, result, channel);

    // convertImageTo32F(result, metaData);
    return result;
}

PaddedImage TiffReader::getFromBuffer(const BoxCoordWithPadding& coord, int bufferIndex) const {
    
    PaddedImage result;
    ImageBuffer& buffer = loadedImageStrips.find(bufferIndex);
    buffer.interactedValue += coord.box.dimensions.width;
    // convertedCoords is what part of the buffer correlates to the requested region coord. The buffer coordinates are offset
    // by the amount of padding included in the buffer, so getting pixel(0,0) of the real image is equivalent to getting pixel(p.before+0,p.before+0) of the buffer
    // where p.before is the padding before the image. The buffer.source.box.position already has the padding offset within. So if a image at (50,50) with padding(20,20)
    // was requested then the buffer.source.box.position is (50, 50) even though it actuall has the image at position (30,30). This is so that it is clear
    // what part of the image might be padding and which part is truly image. This is also why the padding offset might need to be taken into account(usually is the same though)
    BoxCoord convertedCoords{
        coord.box.position - buffer.source.box.position - coord.padding.before + buffer.source.padding.before,
        coord.box.dimensions + coord.padding.before + coord.padding.after
    };
        // the images stored in the ImageBuffer are basically shifted to the bottom right due to the padding
    result.image = std::move(buffer.image.getSubimageCopy(convertedCoords)); 
    if (buffer.interactedValue >= buffer.source.box.dimensions.width){
        loadedImageStrips.deleteIndex(bufferIndex);
    }
    result.padding = coord.padding;
    return result;
}

int TiffReader::getStripIndex(const BoxCoordWithPadding& coord) const {

    for (int i = 0; i < loadedImageStrips.size(); i++){
        if (coord.isWithin(loadedImageStrips.find(i).source)){
            return i;
        }
    }
    return -1;
}

void TiffReader::readStripWithPadding(const BoxCoordWithPadding& coord) const {
    BoxCoord image{CuboidShape{0,0,0}, CuboidShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
    BoxCoord requestedRegion = coord.getBox();
    Padding padding = requestedRegion.cropTo(image);
    padding.before.width = coord.padding.before.width;
    padding.after.width = coord.padding.after.width;
    requestedRegion.dimensions.width = image.dimensions.width;
    requestedRegion.position.width = 0;

    Image3D readImage = managedReader(requestedRegion);
    Preprocessor::padImage(readImage, padding, PaddingType::MIRROR);

    ImageBuffer result;
    result.image = readImage;
    BoxCoordWithPadding source{
        BoxCoord{coord.box.position,
        coord.box.dimensions,},
        coord.padding};
    source.box.position.width = image.position.width;
    source.box.dimensions.width = image.dimensions.width;
    result.source = source;

    loadedImageStrips.push_back(std::move(result));

}

std::optional<PaddedImage> TiffReader::getSubimage(const BoxCoordWithPadding& coord) const {

    std::unique_lock<std::mutex> lock(mutex); //TESTVALUE
    int bufferIndex;
    bufferIndex = getStripIndex(coord);
    if (bufferIndex != -1){
        return getFromBuffer(coord, bufferIndex);
    }
    readStripWithPadding(coord);

    bufferIndex = getStripIndex(coord);
    assert(bufferIndex != -1 && "Still cant find imagebuffer");

    PaddedImage image = getFromBuffer(coord, bufferIndex);
    assert(image.image.getItkImage().IsNotNull());
    return image;
}



const ImageMetaData& TiffReader::getMetaData() const {
    return metaData;
}

void TiffReader::convertImageTo32F(Image3D& image, const ImageMetaData& metaData){
    CuboidShape shape = image.getShape();
    double scale = 1.0 / (metaData.maxSampleValue - metaData.minSampleValue);
    double offset = -metaData.minSampleValue * scale;
    
    int pixelCount = 0;
    int totalPixels = shape.getVolume();
    
    // Use ITK iterator to convert pixel values
    for (auto it = image.begin(); it != image.end(); ++it) {
        float originalValue = *it;
        float convertedValue = static_cast<float>(originalValue * scale + offset);
        *it = convertedValue;
        
        pixelCount++;

    }
}


// Static function for metadata extraction from TIFF file
ImageMetaData TiffReader::extractMetadataFromTiff(TIFF*& tifFile)
{

    ImageMetaData metadatatemp;
    // -------------------------
    // Image description
    // -------------------------
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
    // If slices not found in description, calculate from total images and channels
    if (metadatatemp.slices == 0) {
        int totalDirectories = countTiffDirectories(tifFile);
        
        metadatatemp.slices = totalDirectories / metadatatemp.linChannels;


    }
    // -------------------------
    // TIFF core tags (correct types)
    // -------------------------
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

    // SampleFormat is OPTIONAL — default is UNSIGNED INT
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

    // -------------------------
    // Optional min/max tags (HINTS ONLY)
    // -------------------------
    uint16_t minTag = 0, maxTag = 0;
    bool hasMinTag = TIFFGetField(tifFile, TIFFTAG_MINSAMPLEVALUE, &minTag);
    bool hasMaxTag = TIFFGetField(tifFile, TIFFTAG_MAXSAMPLEVALUE, &maxTag);

    if (hasMinTag && hasMaxTag) {
        metadatatemp.minSampleValue = minTag;
        metadatatemp.maxSampleValue = maxTag;
    }

    // -------------------------
    // DERIVED safe min/max (for scaling without full image scan)
    // -------------------------

    if (sampleFormat == SAMPLEFORMAT_UINT) {
        metadatatemp.minSampleValue = 0.0;
        metadatatemp.maxSampleValue = std::pow(2.0, bps) - 1.0;
     }
    else if (sampleFormat == SAMPLEFORMAT_INT) {
        metadatatemp.minSampleValue = -std::pow(2.0, bps - 1);
        metadatatemp.maxSampleValue =  std::pow(2.0, bps - 1) - 1.0;
    }
    else if (sampleFormat == SAMPLEFORMAT_IEEEFP) {
        // Float TIFF: no implied range
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
    
    // Reset to first directory
    TIFFSetDirectory(tif, 0);
    return count;
}



void TiffReader::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    auto logger = spdlog::get("reader");
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
        logger->warn("TIFF warning (format error): {}", fmt);
        return;
    }

    std::string message;
    message.resize(static_cast<size_t>(required));
    vsnprintf(&message[0], static_cast<size_t>(required) + 1, fmt, ap);

    logger->warn("Tiff Warning Handler: {}", message);
}

void TiffReader::convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, int width, const ImageMetaData& metaData, int channel) {
    for (int x = 0; x < width; x++) {
        if (metaData.bitsPerSample == 8) {
            const uint8_t* data8 = reinterpret_cast<const uint8_t*>(scanlineData);
            rowData[x] = static_cast<float>(data8[x * metaData.samplesPerPixel + channel]);
        } else if (metaData.bitsPerSample == 16) {
            const uint16_t* data16 = reinterpret_cast<const uint16_t*>(scanlineData);
            rowData[x] = static_cast<float>(data16[x * metaData.samplesPerPixel + channel]);
        } else if (metaData.bitsPerSample == 32) {
            const float* data32 = reinterpret_cast<const float*>(scanlineData);
            rowData[x] = data32[x * metaData.samplesPerPixel + channel];
        } else {
            throw TiffReadException("Unsupported bit depth: " + std::to_string(metaData.bitsPerSample) +
                                    " (supported: 8, 16, 32)");
        }

    }
}

