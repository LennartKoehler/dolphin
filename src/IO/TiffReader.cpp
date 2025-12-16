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

#include "IO/TiffReader.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <cstdarg>
#include "deconvolution/Preprocessor.h"
#include "UtlImage.h"
namespace fs = std::filesystem;

// // Default constructor
// HyperstackReader::HyperstackReader() {
//     // Initialize metadata with default values
//     metaData.imageType = "";
//     metaData.filename = "";
//     metaData.description = "";
//     metaData.imageWidth = 0;
//     metaData.imageLength = 0;
//     metaData.frameCount = 0;
//     metaData.resolutionUnit = 0;
//     metaData.samplesPerPixel = 1;
//     metaData.bitsPerSample = 0;
//     metaData.photometricInterpretation = 0;
//     metaData.linChannels = 1;
//     metaData.planarConfig = 0;
//     metaData.totalImages = -1;
//     metaData.slices = 0;
//     metaData.dataType = 0;
//     metaData.xResolution = 0.0f;
//     metaData.yResolution = 0.0f;
// }

// Constructor with filename
TiffReader::TiffReader(std::string filename){
    // Set filename in metadata
    metaData.filename = filename;
    maxBufferMemory_bytes = 999999999; //TESTVALUE
    
    TIFFSetWarningHandler(customTifWarningHandler);
    metaData = extractMetadata(filename);
    currentBufferMemory_bytes = 0;
}

// Destructor
TiffReader::~TiffReader() {
    // Clean up resources if needed
}


// Static method for reading entire TIFF file
Image3D TiffReader::readTiffFile(const std::string& filename) {
    TIFFSetWarningHandler(customTifWarningHandler);
    ImageMetaData metaData = extractMetadata(filename);
    
    Image3D image;
    BoxCoord fullImage{RectangleShape{0,0,0}, RectangleShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
    
    if (!readSubimageFromTiffFile(filename, metaData, fullImage.position.height, fullImage.position.depth, 
                     fullImage.dimensions.height, fullImage.dimensions.depth, fullImage.dimensions.width, image)) {
        std::cerr << "[ERROR] Failed to read TIFF file: " << filename << std::endl;
        return Image3D();
    }
    
    // convertImageTo32F(image, metaData);
    return image;
}

// Static method for extracting metadata
ImageMetaData TiffReader::extractMetadata(const std::string& filename) {
    TIFF* tifFile = TIFFOpen(filename.c_str(), "r");
    if (!tifFile) {
        std::cerr << "[ERROR] Cannot open TIFF file: " << filename << std::endl;
        return ImageMetaData();
    }
    
    ImageMetaData metaData = extractMetadataFromTiff(tifFile);
    metaData.filename = filename;
    
    TIFFClose(tifFile);
    
    if(metaData.slices < 1){
        metaData.slices = metaData.totalImages + 1;
    }
    
    return metaData;
}

bool TiffReader::readSubimageFromTiffFile(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers){
    layers.slices.clear();
    
    TIFFSetWarningHandler(TiffReader::customTifWarningHandler);
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    
    if (!tif) {
        std::cerr << "[ERROR] Cannot open TIFF file: " << filename << std::endl;
        return false;
    }
    

    
    // Validate region shape
    if (height <= 0 || depth <= 0) {
        TIFFClose(tif);
        std::cerr << "[ERROR] Invalid region dimensions: " << height << "x" << depth << std::endl;
        return false;
    }
    

    

    
    // Determine OpenCV data type
    int cvType;
    if (metaData.bitsPerSample == 8) {
        cvType = CV_8UC(metaData.samplesPerPixel);
    } else if (metaData.bitsPerSample == 16) {
        cvType = CV_16UC(metaData.samplesPerPixel);
    } else if (metaData.bitsPerSample == 32) {
        cvType = CV_32FC(metaData.samplesPerPixel);
    } else {
        TIFFClose(tif);
        std::cerr << "[ERROR] Unsupported bit depth: " << metaData.bitsPerSample << std::endl;
        return false;
    }
    
    // Read the specific region using scanline API
    tsize_t scanlineSize = TIFFScanlineSize(tif);
    char* buf = (char*)_TIFFmalloc(scanlineSize);
    if (!buf) {
        TIFFClose(tif);
        std::cerr << "[ERROR] Memory allocation failed for scanline buffer" << std::endl;
        return false;
    }
    
    // Read each directory (z-slice) in the region
    for (uint32_t zIndex = z; zIndex < z + depth; zIndex++) {
        // Set the directory for this z-slice
        if (zIndex > 0 && !TIFFSetDirectory(tif, zIndex)) {
            _TIFFfree(buf);
            TIFFClose(tif);
            std::cerr << "[ERROR] Failed to set directory for z-slice " << zIndex << std::endl;
            return false;
        }
        
        // Create a matrix for this z-slice
        cv::Mat layer(height, width, cvType);
        
        // Read only the required rows (scanlines)
        for (uint32_t yIndex = y; yIndex < y + height; yIndex++) {
            if (TIFFReadScanline(tif, buf, yIndex) == -1) {
                _TIFFfree(buf);
                TIFFClose(tif);
                std::cerr << "[ERROR] Failed to read scanline " << yIndex << " in z-slice " << yIndex << std::endl;
                return false;
            }
            
            // Copy only the required columns
            memcpy(layer.ptr(yIndex - y), buf, scanlineSize);
        }
        
        layers.slices.push_back(layer);
    }
    
    // Clean up
    _TIFFfree(buf);
    TIFFClose(tif);
    //TODO
    // size_t memory = getMemoryForShape(RectangleShape{metaData.imageWidth, height, depth});
    // updateCurrentMemoryBuffer(memory + currentBufferMemory_bytes);
    
    std::cout << "[INFO] Successfully read region: " << height << "x" << depth << std::endl;
    return true;
}



void TiffReader::updateCurrentMemoryBuffer(size_t memory) const {
    // std::unique_lock<std::mutex> lock(mutex);
    currentBufferMemory_bytes = memory;
}

size_t TiffReader::getMemoryForShape(const RectangleShape& shape, const ImageMetaData& metaData){
    // Calculate memory requirement based on metaData's bit depth and samples per pixel
    size_t bytesPerPixel = (metaData.bitsPerSample / 8) * metaData.samplesPerPixel;
    return shape.volume * bytesPerPixel;
}

Image3D TiffReader::managedReader(const BoxCoord& coord) const {
    // std::unique_lock<std::mutex> lock(mutex);
    size_t memorySize = getMemoryForShape(coord.dimensions, metaData);
    // memoryWaiter.wait(lock, [this, memorySize]() {
    //     return currentBufferMemory_bytes + memorySize < maxBufferMemory_bytes;
    // });//TESTVALUE
    Image3D result;
    readSubimageFromTiffFile(metaData.filename, metaData, coord.position.height, coord.position.depth, 
                coord.dimensions.height, coord.dimensions.depth, coord.dimensions.width, result);

    convertImageTo32F(result, metaData);
    return result;
}

PaddedImage TiffReader::getFromBuffer(const BoxCoordWithPadding& coord, int bufferIndex) const {
    
    PaddedImage result;
    ImageBuffer& buffer = loadedImageStrips.find(bufferIndex);
    buffer.interactedValue += coord.box.dimensions.width;
    BoxCoord convertedCoords{
        coord.box.position - buffer.source.box.position,
        coord.box.dimensions + coord.padding.before + coord.padding.after
    };
        // the images stored in the ImageBuffer are basically shifted to the bottom right
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
    BoxCoord image{RectangleShape{0,0,0}, RectangleShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
    BoxCoord requestedRegion = coord.getBox();
    Padding padding = requestedRegion.cropTo(image);
    padding.before.width = coord.padding.before.width;
    padding.after.width = coord.padding.after.width;
    requestedRegion.dimensions.width = image.dimensions.width;
    requestedRegion.position.width = 0;

    Image3D readImage = managedReader(requestedRegion);
    Preprocessor::padImage(readImage, padding, 2);

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
    return getFromBuffer(coord, bufferIndex);
}



const ImageMetaData& TiffReader::getMetaData() const {
    return metaData;
}

void TiffReader::convertImageTo32F(Image3D& layers, const ImageMetaData& metaData){
    int i = 0;
    for (auto& layer : layers.slices) {
        int type = CV_MAKETYPE(CV_32F, layer.channels());
        layer.convertTo(layer, type, 1 / (metaData.maxSampleValue - metaData.minSampleValue), - metaData.minSampleValue * (1 / (metaData.maxSampleValue - metaData.minSampleValue)));

        std::cout << "\r[STATUS] Layer " << i << "/" << layers.slices.size() - 1 << " in 32F converted"
                  << " ";
        std::flush(std::cout);
        i++;
    }
    std::cout << std::to_string(layers.slices[1].at<float>(1, 1)) << std::endl;
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
void TiffReader::createImage3D(const Image3D& layers, std::vector<Channel>& channels) {
    if(metaData.linChannels > 0){
        std::vector<Image3D> channelData(metaData.linChannels);
        int c = 0;
        int z = 0;
        int multichannel_z = ((metaData.totalImages + 1) / metaData.linChannels);
        bool success = false;
        for(auto& singleLayer : layers.slices){
            channelData[c].slices.push_back(singleLayer);
            c++;
            if(c > metaData.linChannels-1){
                c = 0;
                z++;
            }
            if(multichannel_z == z) {
                std::cout <<"[INFO] " << metaData.filename << " converted to multichannel" << std::endl;
                success = true;
            }
        }
        if(!success){
            std::cout << "[ERROR] "<< metaData.filename << "(Layers: " << std::to_string(layers.slices.size()) << ") could not converted to multichannel, Layers: " << std::to_string(z) << std::endl;
        }

        //create channels with Image3D data
        int id = 0;
        for(auto& imageData : channelData){
            Channel channel;
            channel.image = imageData;
            channel.id = id;
            channels.push_back(channel);
            id++;
        }

    }else{
        Image3D imageLayers;
        imageLayers.slices = layers.slices;
        Channel channel;
        channel.image = imageLayers;
        channel.id = 0;
        channels.push_back(channel);
    }
}

void TiffReader::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    // Ignoriere alle Warnungen oder filtere nach bestimmten Tags
    // Beispiel: printf(fmt, ap); // Um die Warnungen anzuzeigen
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
//         std::cerr << metaData.bitsPerSample << "[ERROR] Unsupported bit depth." << std::endl;
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
//             std::cerr << "[ERROR] Memory allocation failed for buffer." << std::endl;
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
//         //std::cout << "Layer  " << depth << " successfully read" << std::endl;
//     } while (TIFFReadDirectory(tifFile));

//     //TODO debug
//     std::cout<< "[INFO] Read in " << layers.slices.size() << " layers"<< std::endl;

//     return true;
// }
// // Static function for metadata extraction from directory
// bool TiffReader::extractImageDataFromDirectory(const std::string& directoryPath, ImageMetaData& metaData) {
//     fs::path dirPath(directoryPath);
//     if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
//         std::cerr << "[ERROR] Specified path is not a directory or does not exist: " << directoryPath << std::endl;
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

//     std::cerr << "[ERROR] No TIFF files found in directory: " << directoryPath << std::endl;
//     return false;
// }

// // Static function for reading layers from directory
// bool TiffReader::readLayersFromDirectory(const std::string& directoryPath, Image3D& layers, ImageMetaData& metaData) {
//     fs::path dirPath(directoryPath);
//     if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
//         std::cerr << "[ERROR] Specified path is not a directory or does not exist: " << directoryPath << std::endl;
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
//             std::cerr << "[ERROR] Could not open TIFF file: " << filePath.string() << std::endl;
//             return false;
//         }
//     }

//     std::cout << "[INFO] Read in " << layers.slices.size() << " layers from directory" << std::endl;
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


