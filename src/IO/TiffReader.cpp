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
    metaData = extractMetadataStatic(filename);
    currentBufferMemory_bytes = 0;
    tif = TIFFOpen(filename.c_str(), "r");
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
Image3D TiffReader::readTiffFile(const std::string& filename, int channel) {
    TIFFSetWarningHandler(customTifWarningHandler);
    ImageMetaData metaData = extractMetadataStatic(filename);
    
    Image3D image;
    BoxCoord fullImage{CuboidShape{0,0,0}, CuboidShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
    
    if (!readSubimageFromTiffFileStatic(filename, metaData, fullImage.position.height, fullImage.position.depth, 
                     fullImage.dimensions.height, fullImage.dimensions.depth, fullImage.dimensions.width, image, channel)) {
        spdlog::error(" Failed to read TIFF file: {}", filename);
        return Image3D();
    }
    
    // convertImageTo32F(image, metaData);
    return image;
}

// Static method for extracting metadata
ImageMetaData TiffReader::extractMetadataStatic(const std::string& filename) {
    TIFF* tifFile = TIFFOpen(filename.c_str(), "r");
    if (!tifFile) {
        spdlog::error("Cannot open TIFF file to read metadata: {}", filename);
        throw; 
    }
    
    ImageMetaData metaData = extractMetadataFromTiff(tifFile);
    metaData.filename = filename;
    
    TIFFClose(tifFile);
    
    if(metaData.slices < 1){
        metaData.slices = metaData.totalImages + 1;
    }
    
    return metaData;
}


ImageMetaData TiffReader::extractMetadata(){
    return extractMetadataStatic(metaData.filename);
}

bool TiffReader::readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& image, int channel){
     
    TIFFSetWarningHandler(TiffReader::customTifWarningHandler);
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    
    bool succ = readSubimageFromTiffFile(tif, metaData, y, z, height, depth, width, image, channel);
    if (succ == false){
        return false;
    }
    TIFFClose(tif);
    convertImageTo32F(image, metaData);
    
    return true;
}




// Non-static method using member TIFF* variable
bool TiffReader::readSubimageFromTiffFile(TIFF* tiffile, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& image, int channel) {

    if (!tiffile) {
        spdlog::error("TIFF file is not open");
        return false;
    }
    // there are different ways to store multiple channels, either by interleaved directories, or by multiple samples per pixels
    // i dotn think both are used together
    int ifdchannel = 0; // image file directory
    int sppchannel = 0; // samples per pixel
    if (metaData.linChannels > 1){
        ifdchannel = channel - 1; // channel is 1 based
        if (ifdchannel > metaData.linChannels - 1){
            spdlog::error(" Specified channel {} larger than maximum number of image file directories in image: {}", channel, metaData.linChannels);
            return false;
        }
    }
    else if (metaData.samplesPerPixel > 1){
        sppchannel = channel - 1; // channel is 1 based
        if (sppchannel > metaData.samplesPerPixel - 1){
            spdlog::error(" Specified channel {} larger than maximum number of samples per pixel in image: {}", channel, metaData.samplesPerPixel);
            return false;
        }
    }
    
    // Validate region shape
    if (height <= 0 || depth <= 0) {
        spdlog::error(" Invalid region dimensions: {} x {}", height, depth);
        return false;
    }
    
    // Create ITK image with the specified dimensions
    CuboidShape imageShape(width, height, depth);
    image = Image3D(imageShape);
    
    // Read the specific region using scanline API
    tsize_t scanlineSize = TIFFScanlineSize(tiffile);
    char* buf = (char*)_TIFFmalloc(scanlineSize);
    if (!buf) {
        spdlog::error("Memory allocation failed for scanline buffer");
        return false;
    }
    
    // Create a temporary buffer for conversion
    std::vector<float> rowData(width);
    
    // Read each directory (z-slice) in the region
    for (uint32_t zIndex = z; zIndex < z + depth; zIndex++) {
        // Always set the directory for this z-slice (including z=0)

        int zIndexChannel = (zIndex * metaData.linChannels) + ifdchannel;
        if (!TIFFSetDirectory(tiffile, zIndexChannel)) {
            _TIFFfree(buf);
            spdlog::error(" Failed to set directory for z-slice ", zIndex);
            return false;
        }
        
        // Read only the required rows (scanlines)
        for (uint32_t yIndex = y; yIndex < y + height; yIndex++) {
            if (TIFFReadScanline(tiffile, buf, yIndex) == -1) {
                _TIFFfree(buf);
                spdlog::error(" Failed to read scanline {} in z-slice {}", yIndex, zIndex);
                return false;
            }
            
            // Convert scanline data based on bit depth and set into ITK image
            convertScanlineToFloat(buf, rowData, width, metaData, sppchannel);
            
            // Set the row data into the ITK image
            image.setRow(yIndex - y, zIndex - z, rowData.data());

        }
    }
    
    _TIFFfree(buf);
    
    spdlog::info("Successfully read strip ({}): ({},{}) {}x{}", metaData.filename, y, z, height, depth);
    return true;
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

#include "dolphin/IO/TiffWriter.h"
PaddedImage TiffReader::getFromBuffer(const BoxCoordWithPadding& coord, int bufferIndex) const {
    
    PaddedImage result;
    ImageBuffer& buffer = loadedImageStrips.find(bufferIndex);
    buffer.interactedValue += coord.box.dimensions.width;
    BoxCoord convertedCoords{
        coord.box.position - buffer.source.box.position,
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

    loadedImageStrips.push_back(result);
}

PaddedImage TiffReader::getSubimage(const BoxCoordWithPadding& coord) const {

    std::unique_lock<std::mutex> lock(mutex); //TESTVALUE
    int bufferIndex;
    bufferIndex = getStripIndex(coord);
    if (bufferIndex != -1){
        return getFromBuffer(coord, bufferIndex);
    }
    readStripWithPadding(coord);

    bufferIndex = getStripIndex(coord);
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
    spdlog::info("");
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
    // Ignoriere alle Warnungen oder filtere nach bestimmten Tags
    // Beispiel: printf(fmt, ap); // Um die Warnungen anzuzeigen
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
            spdlog::error(" Unsupported bit depth: ", metaData.bitsPerSample);
            // Fill with zeros as fallback
            std::fill(rowData.begin(), rowData.end(), 0.0f);
        }
    }
}

// std::string TiffReader::getFilename(const std::string& path) {
//     size_t pos = path.find_last_of("/\\");
//     if (pos == std::string::npos) {
//         return path; // No directory separator found, return whole string
//     }
//     return path.substr(pos + 1);
// }


// // Static function for reading layers from TIFF file
// bool TiffReader::readLayersFromTifFile(TIFF*& tifFile, Image3D& layers, const ImageMetaData& metaData) {
//     if (metaData.bitsPerSample == 8) {
//         // dataType will be set in the loop based on samplesPerPixel
//     } else if (metaData.bitsPerSample == 16) {
//         // dataType will be set in the loop based on samplesPerPixel
//     } else if (metaData.bitsPerSample == 32) {
//         // dataType will be set in the loop based on samplesPerPixel
//     } else {
//         spdlog::error("Unsupported bit depth.");
//         return false;
//     }
    
//     do {
//         // Note: We can't modify metaData directly as it's const, so we need to track this differently
//         static int totalImages = 0;
//         totalImages++;
        
//         uint32_t width, height;
//         TIFFGetField(tifFile, TIFFTAG_IMAGEWIDTH, &width);
//         TIFFGetField(tifFile, TIFFTAG_IMAGELENGTH, &height);
//         tsize_t scanlineSize = TIFFScanlineSize(tifFile);

//         int dataType;
//         if (metaData.bitsPerSample == 8) {
//             dataType = CV_8UC(metaData.samplesPerPixel);
//         } else if (metaData.bitsPerSample == 16) {
//             dataType = CV_16UC(metaData.samplesPerPixel);
//         } else if (metaData.bitsPerSample == 32) {
//             dataType = CV_32FC(metaData.samplesPerPixel);
//         }

//         cv::Mat layer = cv::Mat(height, width, dataType);

//         char* buf;
//         uint32_t row;
//         buf = (char *)_TIFFmalloc(scanlineSize);
//         if (!buf) {
//             spdlog::error("Memory allocation failed for buffer.");
//             TIFFClose(tifFile);
//             return false;
//         }

//         for (row = 0; row < height; row++) {
//             TIFFReadScanline(tifFile, buf, row);
//             memcpy(layer.ptr(row), buf, scanlineSize);
//         }

//         layers.slices.push_back(layer);

//         _TIFFfree(buf);

//         //TODO debug option
//         //spdlog::info("Layer  {} successfully read", depth);
//     } while (TIFFReadDirectory(tifFile));

//     //TODO debug
//     spdlog::info("Read in {} layers", layers.slices.size());

//     return true;
// }
// // Static function for metadata extraction from directory
// bool TiffReader::extractImageDataFromDirectory(const std::string& directoryPath, ImageMetaData& metaData) {
//     fs::path dirPath(directoryPath);
//     if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
//         spdlog::error(" Specified path is not a directory or does not exist: ", directoryPath);
//         return false;
//     }

//     // Initialize metadata with default values
//     metaData.imageType = "";
//     metaData.description = "";
//     metaData.imageWidth = 0;
//     metaData.imageLength = 0;
//     metaData.frameCount = 0;
//     metaData.resolutionUnit = 0;
//     metaData.samplesPerPixel = 0;
//     metaData.bitsPerSample = 0;
//     metaData.photometricInterpretation = 0;
//     metaData.linChannels = 1;
//     metaData.planarConfig = 0;
//     metaData.totalImages = -1;
//     metaData.slices = 0;
//     metaData.dataType = 0;
//     metaData.xResolution = 0.0f;
//     metaData.yResolution = 0.0f;

//     // Try to find the first TIFF file in the directory to extract metadata
//     for (const auto& entry : fs::directory_iterator(dirPath)) {
//         if (entry.is_regular_file()) {
//             std::string extension = entry.path().extension().string();
//             std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
//             if (extension == ".tif" || extension == ".tiff") {
//                 TIFF* tifFile = TIFFOpen(entry.path().string().c_str(), "r");
//                 if (tifFile) {
//                     bool success = extractMetadata(tifFile, metaData);
//                     TIFFClose(tifFile);
//                     if (success) {
//                         metaData.filename = directoryPath;
//                         return true;
//                     }
//                 }
//             }
//         }
//     }

//     spdlog::error(" No TIFF files found in directory: ", directoryPath);
//     return false;
// }

// // Static function for reading layers from directory
// bool TiffReader::readLayersFromDirectory(const std::string& directoryPath, Image3D& layers, ImageMetaData& metaData) {
//     fs::path dirPath(directoryPath);
//     if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
//         spdlog::error(" Specified path is not a directory or does not exist: ", directoryPath);
//         return false;
//     }

//     // Collect all TIFF files in the directory
//     std::vector<fs::path> tiffFiles;
//     for (const auto& entry : fs::directory_iterator(dirPath)) {
//         if (entry.is_regular_file()) {
//             std::string extension = entry.path().extension().string();
//             std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
//             if (extension == ".tif" || extension == ".tiff") {
//                 tiffFiles.push_back(entry.path());
//             }
//         }
//     }

//     // Sort files to ensure consistent order
//     std::sort(tiffFiles.begin(), tiffFiles.end());

//     // Read each TIFF file
//     for (const auto& filePath : tiffFiles) {
//         TIFF* tifFile = TIFFOpen(filePath.string().c_str(), "r");
//         if (tifFile) {
//             if (!readLayersFromTifFile(tifFile, layers, metaData)) {
//                 TIFFClose(tifFile);
//                 return false;
//             }
//             TIFFClose(tifFile);
//         } else {
//             spdlog::error(" Could not open TIFF file: ", filePath.string());
//             return false;
//         }
//     }

//     spdlog::info("Read in {} layers from directory", layers.slices.size());
//     return true;
// }

// bool TiffReader::readFromTifDir(const std::string& directoryPath, std::vector<Channel>& channels, Image3D& layers) {
//     // Clear output containers
//     channels.clear();
//     layers.slices.clear();

//     // Extract metadata from directory using static method
//     if (!extractImageDataFromDirectory(directoryPath, metaData)) {
//         return false;
//     }

//     // Read layers from directory using static method
//     if (!readLayersFromDirectory(directoryPath, layers, metaData)) {
//         return false;
//     }

//     // Convert layers to 32F format
//     convertImageTo32F(layers);

//     // Create channel images from layers
//     createImage3D(layers, channels);

//     return true;
// }


