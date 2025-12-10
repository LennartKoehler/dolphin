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
    // Create temporary containers for reading
    
    TIFFSetWarningHandler(customTifWarningHandler);
    TIFF* tifOriginalFile = TIFFOpen(filename.c_str(), "r");
    extractMetadata(tifOriginalFile, metaData);


    TIFFClose(tifOriginalFile);

    if(metaData.slices < 1){
        metaData.slices = metaData.totalImages+1;
    }
}

// Destructor
TiffReader::~TiffReader() {
    // Clean up resources if needed
}


bool TiffReader::readFromFile(std::string filename, int y, int z, int height, int depth, Image3D& layers) const {
    layers.slices.clear();
    
    TIFFSetWarningHandler(TiffReader::customTifWarningHandler);
    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    
    if (!tif) {
        std::cerr << "[ERROR] Cannot open TIFF file: " << filename << std::endl;
        return false;
    }
    
    // Get TIFF dimensions
    uint32_t imageWidth, imageLength;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imageWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageLength);
    
    // Validate region shape
    if (height <= 0 || depth <= 0) {
        TIFFClose(tif);
        std::cerr << "[ERROR] Invalid region dimensions: " << height << "x" << depth << std::endl;
        return false;
    }
    
    // Check if region coordinates are within image bounds
    if (height > imageLength) {
        TIFFClose(tif);
        std::cerr << "[ERROR] Region dimensions exceed image dimensions" << std::endl;
        std::cerr << "Region: " << height << " at (" << 0 << "," << 0 << ")" << std::endl;
        std::cerr << "Image: " << imageWidth << "x" << imageLength << std::endl;
        return false;
    }
    
    // Get image metadata for data type
    uint16_t bitsPerSample, samplesPerPixel;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    
    // Determine OpenCV data type
    int cvType;
    if (bitsPerSample == 8) {
        cvType = CV_8UC(samplesPerPixel);
    } else if (bitsPerSample == 16) {
        cvType = CV_16UC(samplesPerPixel);
    } else if (bitsPerSample == 32) {
        cvType = CV_32FC(samplesPerPixel);
    } else {
        TIFFClose(tif);
        std::cerr << "[ERROR] Unsupported bit depth: " << bitsPerSample << std::endl;
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
        cv::Mat layer(height, metaData.imageWidth, cvType);
        
        // Read only the required rows (scanlines)
        for (uint32_t yIndex = y; yIndex < z + height; yIndex++) {
            if (TIFFReadScanline(tif, buf, yIndex) == -1) {
                _TIFFfree(buf);
                TIFFClose(tif);
                std::cerr << "[ERROR] Failed to read scanline " << yIndex << " in z-slice " << yIndex << std::endl;
                return false;
            }
            
            // Copy only the required columns
            memcpy(layer.ptr(yIndex), buf, metaData.imageWidth * samplesPerPixel * (bitsPerSample / 8));
        }
        
        layers.slices.push_back(layer);
    }
    
    // Clean up
    _TIFFfree(buf);
    TIFFClose(tif);

    size_t memory = getMemoryForShape(RectangleShape{metaData.imageWidth, height, depth});
    updateCurrentMemoryBuffer(memory + currentBufferMemory_bytes);
    
    std::cout << "[INFO] Successfully read region: " << height << "x" << depth << std::endl;
    return true;
}

void TiffReader::updateCurrentMemoryBuffer(size_t memory) const {
    std::unique_lock<std::mutex> lock(mutex);
    currentBufferMemory_bytes = memory;
}

size_t TiffReader::getMemoryForShape(const RectangleShape& shape) const {
    // Calculate memory requirement based on metaData's bit depth and samples per pixel
    size_t bytesPerPixel = (metaData.bitsPerSample / 8) * metaData.samplesPerPixel;
    return shape.volume * bytesPerPixel;
}

Image3D TiffReader::managedReader(int y, int z, int height, int depth) const {
    std::unique_lock<std::mutex> lock(mutex);
    size_t memorySize = getMemoryForShape(RectangleShape{metaData.imageWidth, height, depth});
    memoryWaiter.wait(lock, [this, memorySize]() {
        return currentBufferMemory_bytes + memorySize < maxBufferMemory_bytes;
    });
    Image3D result;
    readFromFile(metaData.filename, y, z, height, depth, result);

    convertImageTo32F(result);
    return result;
}

Image3D TiffReader::managedReader(const BoxCoord& coord) const {
    return managedReader(coord.position.height, coord.position.depth, coord.dimensions.height, coord.dimensions.depth);
}

PaddedImage TiffReader::getFromBuffer(const BoxCoordWithPadding& coord, int bufferIndex) const {
    PaddedImage result;
    BoxCoord convertedCoords{
        RectangleShape(coord.box.position),
        coord.box.dimensions + coord.padding.before + coord.padding.after
    };
        // the images stored in the ImageBuffer are basically shifted to the bottom right
    result.image = std::move(loadedImageStrips[bufferIndex].image.getSubimageCopy(convertedCoords)); 
    result.padding = coord.padding;
    return result;
}

int TiffReader::getBufferIndex(const BoxCoordWithPadding& coord) const {
    for (int i = 0; i < loadedImageStrips.size(); i++){
        if (coord.isWithin(loadedImageStrips[i].source)){
            return i;
        }
    }
    return -1;
}
void TiffReader::readStripWithPadding(const BoxCoordWithPadding& coord) const {
    BoxCoord image{RectangleShape{0,0,0}, RectangleShape{metaData.imageWidth, metaData.imageLength, metaData.slices}};
    BoxCoord requestedRegion{coord.box.position - coord.padding.before, coord.box.position + coord.box.dimensions + coord.padding.after};
    Padding padding = requestedRegion.cropTo(image);
    Image3D readImage = managedReader(requestedRegion);
    Preprocessor::padImage(readImage, padding, 0);
    ImageBuffer result;
    result.image = readImage;
    BoxCoordWithPadding source{coord.box, padding};
    source.box.dimensions.width = metaData.imageWidth;
    source.box.position.width = 0;
    result.source = source;
    loadedImageStrips.push_back(result);
}

PaddedImage TiffReader::getSubimage(const BoxCoordWithPadding& coord) const {
    int bufferIndex;
    bufferIndex = getBufferIndex(coord);
    if (bufferIndex != -1){
        return getFromBuffer(coord, bufferIndex);
    }
    readStripWithPadding(coord);

    bufferIndex = getBufferIndex(coord);
    return getFromBuffer(coord, bufferIndex);
    
}



const ImageMetaData& TiffReader::getMetaData() const {
    return metaData;
}

// Helper methods
void TiffReader::convertImageTo32F(Image3D& layers) const {
    double globalMin, globalMax;
    
    if (!layers.slices.empty()) {
        globalMin = layers.slices[0].at<uchar>(0, 0);
        globalMax = layers.slices[0].at<uchar>(0, 0);
        
        for (const auto& layer : layers.slices) {
            if (layer.depth() == CV_8U) {
                for (int y = 0; y < layer.rows; ++y) {
                    for (int x = 0; x < layer.cols; ++x) {
                        uchar val = layer.at<uchar>(y, x);
                        globalMin = std::min(globalMin, (double)val);
                        globalMax = std::max(globalMax, (double)val);
                    }
                }
            } else if (layer.depth() == CV_16U) {
                for (int y = 0; y < layer.rows; ++y) {
                    for (int x = 0; x < layer.cols; ++x) {
                        ushort val = layer.at<ushort>(y, x);
                        globalMin = std::min(globalMin, (double)val);
                        globalMax = std::max(globalMax, (double)val);
                    }
                }
            }
        }
    }

    int i = 0;
    for (auto& layer : layers.slices) {
        int type = CV_MAKETYPE(CV_32F, layer.channels());
        layer.convertTo(layer, type, 1 / (globalMax - globalMin), -globalMin * (1 / (globalMax - globalMin)));

        std::cout << "\r[STATUS] Layer " << i << "/" << layers.slices.size() - 1 << " in 32F converted"
                  << " ";
        std::flush(std::cout);
        i++;
    }
    if (!layers.slices.empty()) {
        std::cout << layers.slices[1].at<float>(1, 1) << std::endl;
    }

    // Note: We don't modify metaData here since this method should be const
    // The original code modified metaData.bitsPerSample = 32 and metaData.dataType = CV_32F
    // but this breaks const correctness. The metadata should reflect the original file,
    // not the converted data.
}


// Main methods
bool TiffReader::readFromTifFile(std::string filename, std::vector<Channel>& channels, Image3D& layers) {

    
    channels.clear();
    layers.slices.clear();
    
    //Reading File
    TIFFSetWarningHandler(TiffReader::customTifWarningHandler);
    TIFF* tifOriginalFile = TIFFOpen(filename.c_str(), "r");
    extractMetadata(tifOriginalFile, metaData);

    //Read Layers
    if (!readLayersFromTifFile(tifOriginalFile, layers, metaData)) {
        TIFFClose(tifOriginalFile);
        return false;
    }
    TIFFClose(tifOriginalFile);

    //Converting Layers to 32F
    convertImageTo32F(layers);

    //Creating Channel Images
    createImage3D(layers, channels);

    if(metaData.slices < 1){
        metaData.slices = metaData.totalImages+1;
    }

    std::cout << "[INFO] Read in metadata successful" <<std::endl;

    return true;
}
// Static function for metadata extraction from TIFF file
bool TiffReader::extractMetadata(TIFF*& tifFile, ImageMetaData& metaData) {
    if (!tifFile) {
        std::cerr << "[ERROR] Cannot open TIFF file" << std::endl;
        return false;
    }else{
        //Extract Data
        char* img_description;
        if (TIFFGetField(tifFile, TIFFTAG_IMAGEDESCRIPTION, &img_description)) {
            std::string desc(img_description);
            std::istringstream iss(desc);
            std::string line;
            while (getline(iss, line)) {
                if (line.find("channels=") != std::string::npos) {
                    metaData.linChannels = std::stoi(line.substr(line.find("=") + 1));
                }
                if (line.find("slices=") != std::string::npos) {
                    metaData.slices = std::stoi(line.substr(line.find("=") + 1));
                }
            }
            metaData.description = desc;
        } else {
            std::cout << "[INFO] No image description" << std::endl;
        }

        // Read TIFF-Tags
        int tempWidth, tempLength, tempResUnit, tempspp, temppi, tempbps, tempfc, temppc;
        float tempXRes, tempYRes;
        if (TIFFGetField(tifFile, TIFFTAG_IMAGEWIDTH, &tempWidth))
            metaData.imageWidth = tempWidth;
        if (TIFFGetField(tifFile, TIFFTAG_IMAGELENGTH, &tempLength))
            metaData.imageLength = tempLength;
        if (TIFFGetField(tifFile, TIFFTAG_RESOLUTIONUNIT, &tempResUnit))
            metaData.resolutionUnit = tempResUnit;
        if (TIFFGetField(tifFile, TIFFTAG_XRESOLUTION, &tempXRes) && tempXRes > 0)
            metaData.xResolution = tempXRes;
        if (TIFFGetField(tifFile, TIFFTAG_YRESOLUTION, &tempYRes) && tempYRes > 0)
            metaData.yResolution = tempYRes;
        if (TIFFGetField(tifFile, TIFFTAG_SAMPLESPERPIXEL, &tempspp) && tempspp > 0)
            metaData.samplesPerPixel = tempspp;
        if (TIFFGetField(tifFile, TIFFTAG_PHOTOMETRIC, &temppi))
            metaData.photometricInterpretation = temppi;
        if (TIFFGetField(tifFile, TIFFTAG_BITSPERSAMPLE, &tempbps))
            metaData.bitsPerSample = tempbps;
        if (TIFFGetField(tifFile, TIFFTAG_FRAMECOUNT, &tempfc))
            metaData.frameCount = tempfc;
        if (TIFFGetField(tifFile, TIFFTAG_PLANARCONFIG, &temppc)) {
            metaData.planarConfig = temppc;
        }

        return true;
    }
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

std::string TiffReader::getFilename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path; // No directory separator found, return whole string
    }
    return path.substr(pos + 1);
}


// Static function for reading layers from TIFF file
bool TiffReader::readLayersFromTifFile(TIFF*& tifFile, Image3D& layers, const ImageMetaData& metaData) {
    if (metaData.bitsPerSample == 8) {
        // dataType will be set in the loop based on samplesPerPixel
    } else if (metaData.bitsPerSample == 16) {
        // dataType will be set in the loop based on samplesPerPixel
    } else if (metaData.bitsPerSample == 32) {
        // dataType will be set in the loop based on samplesPerPixel
    } else {
        std::cerr << metaData.bitsPerSample << "[ERROR] Unsupported bit depth." << std::endl;
        return false;
    }
    
    do {
        // Note: We can't modify metaData directly as it's const, so we need to track this differently
        static int totalImages = 0;
        totalImages++;
        
        uint32_t width, height;
        TIFFGetField(tifFile, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tifFile, TIFFTAG_IMAGELENGTH, &height);
        tsize_t scanlineSize = TIFFScanlineSize(tifFile);

        int dataType;
        if (metaData.bitsPerSample == 8) {
            dataType = CV_8UC(metaData.samplesPerPixel);
        } else if (metaData.bitsPerSample == 16) {
            dataType = CV_16UC(metaData.samplesPerPixel);
        } else if (metaData.bitsPerSample == 32) {
            dataType = CV_32FC(metaData.samplesPerPixel);
        }

        cv::Mat layer = cv::Mat(height, width, dataType);

        char* buf;
        uint32_t row;
        buf = (char *)_TIFFmalloc(scanlineSize);
        if (!buf) {
            std::cerr << "[ERROR] Memory allocation failed for buffer." << std::endl;
            TIFFClose(tifFile);
            return false;
        }

        for (row = 0; row < height; row++) {
            TIFFReadScanline(tifFile, buf, row);
            memcpy(layer.ptr(row), buf, scanlineSize);
        }

        layers.slices.push_back(layer);

        _TIFFfree(buf);

        //TODO debug option
        //std::cout << "Layer  " << depth << " successfully read" << std::endl;
    } while (TIFFReadDirectory(tifFile));

    //TODO debug
    std::cout<< "[INFO] Read in " << layers.slices.size() << " layers"<< std::endl;

    return true;
}
// Static function for metadata extraction from directory
bool TiffReader::extractImageDataFromDirectory(const std::string& directoryPath, ImageMetaData& metaData) {
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "[ERROR] Specified path is not a directory or does not exist: " << directoryPath << std::endl;
        return false;
    }

    // Initialize metadata with default values
    metaData.imageType = "";
    metaData.description = "";
    metaData.imageWidth = 0;
    metaData.imageLength = 0;
    metaData.frameCount = 0;
    metaData.resolutionUnit = 0;
    metaData.samplesPerPixel = 0;
    metaData.bitsPerSample = 0;
    metaData.photometricInterpretation = 0;
    metaData.linChannels = 1;
    metaData.planarConfig = 0;
    metaData.totalImages = -1;
    metaData.slices = 0;
    metaData.dataType = 0;
    metaData.xResolution = 0.0f;
    metaData.yResolution = 0.0f;

    // Try to find the first TIFF file in the directory to extract metadata
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".tif" || extension == ".tiff") {
                TIFF* tifFile = TIFFOpen(entry.path().string().c_str(), "r");
                if (tifFile) {
                    bool success = extractMetadata(tifFile, metaData);
                    TIFFClose(tifFile);
                    if (success) {
                        metaData.filename = directoryPath;
                        return true;
                    }
                }
            }
        }
    }

    std::cerr << "[ERROR] No TIFF files found in directory: " << directoryPath << std::endl;
    return false;
}

// Static function for reading layers from directory
bool TiffReader::readLayersFromDirectory(const std::string& directoryPath, Image3D& layers, ImageMetaData& metaData) {
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "[ERROR] Specified path is not a directory or does not exist: " << directoryPath << std::endl;
        return false;
    }

    // Collect all TIFF files in the directory
    std::vector<fs::path> tiffFiles;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".tif" || extension == ".tiff") {
                tiffFiles.push_back(entry.path());
            }
        }
    }

    // Sort files to ensure consistent order
    std::sort(tiffFiles.begin(), tiffFiles.end());

    // Read each TIFF file
    for (const auto& filePath : tiffFiles) {
        TIFF* tifFile = TIFFOpen(filePath.string().c_str(), "r");
        if (tifFile) {
            if (!readLayersFromTifFile(tifFile, layers, metaData)) {
                TIFFClose(tifFile);
                return false;
            }
            TIFFClose(tifFile);
        } else {
            std::cerr << "[ERROR] Could not open TIFF file: " << filePath.string() << std::endl;
            return false;
        }
    }

    std::cout << "[INFO] Read in " << layers.slices.size() << " layers from directory" << std::endl;
    return true;
}

bool TiffReader::readFromTifDir(const std::string& directoryPath, std::vector<Channel>& channels, Image3D& layers) {
    // Clear output containers
    channels.clear();
    layers.slices.clear();

    // Extract metadata from directory using static method
    if (!extractImageDataFromDirectory(directoryPath, metaData)) {
        return false;
    }

    // Read layers from directory using static method
    if (!readLayersFromDirectory(directoryPath, layers, metaData)) {
        return false;
    }

    // Convert layers to 32F format
    convertImageTo32F(layers);

    // Create channel images from layers
    createImage3D(layers, channels);

    return true;
}


