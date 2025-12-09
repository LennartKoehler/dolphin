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

#include "HyperstackImage.h"
#include "HyperstackReader.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "UtlIO.h"
#include "psf/PSF.h"
#include <filesystem>
#include <fstream>


namespace fs = std::filesystem;

bool Hyperstack::readFromTifFile(const char *filename) {
    //METADATA
    std::string imageType = "";
    std::string name = "";
    std::string description = "";
    int imageWidth, imageLength = 0;
    int frameCount = 0;
    uint16_t resolutionUnit = 0;
    uint16_t samplesPerPixel = 1; //num of channels
    uint16_t bitsPerSample = 0;//bit depth
    uint16_t photometricInterpretation = 0;
    int linChannels = 1;//in Description (linearized channels)
    uint16_t planarConfig = 0;
    int totalImages = -1;
    int slices = 0;
    int dataType = 0; //calculated
    float xResolution, yResolution = 0.0f;
    std::vector<cv::Mat> layers;
    
     //Reading File
    TIFFSetWarningHandler(UtlIO::customTifWarningHandler);
    TIFF* tifOriginalFile = TIFFOpen(filename, "r");
    UtlIO::extractData(tifOriginalFile, name, description, filename, linChannels, slices, imageWidth, imageLength, resolutionUnit, xResolution, yResolution, samplesPerPixel, photometricInterpretation, bitsPerSample, frameCount, planarConfig, totalImages);

    //Read Layers
    UtlIO::readLayers(layers, totalImages, dataType, bitsPerSample, samplesPerPixel, tifOriginalFile);
    TIFFClose(tifOriginalFile);

    //Converting Layers to 32F
    UtlIO::convertImageTo32F(layers, dataType, bitsPerSample);

    //Creating Channel Images
    Image3D imageLayers;
    std::vector<Channel> channels;
    UtlIO::createImage3D(channels, imageLayers, linChannels, totalImages, name, layers);
    this->channels = channels;

    //Create Metadata
    ImageMetaData fileMetaData;
    fileMetaData.imageType = imageType;
    fileMetaData.filename = name;
    fileMetaData.description = description;
    fileMetaData.imageWidth = imageWidth;
    fileMetaData.imageLength = imageLength;
    fileMetaData.frameCount = frameCount;
    fileMetaData.resolutionUnit = resolutionUnit;
    fileMetaData.samplesPerPixel = samplesPerPixel; //num of channels
    fileMetaData.bitsPerSample = bitsPerSample;//bit depth
    fileMetaData.photometricInterpretation = photometricInterpretation;
    fileMetaData.linChannels = linChannels;//in Description (linearized channels)
    fileMetaData.planarConfig = planarConfig;
    fileMetaData.totalImages = totalImages;
    if(slices < 1){
        slices = totalImages+1;
    }
    fileMetaData.slices = slices;
    fileMetaData.dataType = dataType; //calculated
    fileMetaData.xResolution = xResolution;
    fileMetaData.yResolution = yResolution;

    this->metaData = fileMetaData;
    std::cout << "[INFO] Read in metadata successful" <<std::endl;

    return true;
}


bool Hyperstack::isValid(){
    return !channels.empty();
}
bool Hyperstack::readFromTifDir(const std::string& directoryPath) {
    HyperstackReader reader;
    if (reader.readFromTifDir(directoryPath)) {
        this->channels = reader.getChannels();
        this->metaData = reader.getMetaData();
        return true;
    }
    return false;
}

bool Hyperstack::saveAsTifFile(const std::string &directoryPath) const {
    std::vector<std::vector<cv::Mat>> channel_vec;
    for (auto& channel : this->channels) {
        channel_vec.push_back(channel.image.slices);
    }
    std::filesystem::create_directories(std::filesystem::path(directoryPath).parent_path());
    if (!std::filesystem::exists(directoryPath)) {
        std::ofstream file(directoryPath);
        if (file.is_open()) {
            file.close();
        }
    }
    TIFF* out = TIFFOpen(directoryPath.c_str(), "w");
    if (!out) {
        std::cerr << "[ERROR] Cannot open TIFF file for writing: " << directoryPath.c_str() << std::endl;
        return false;
    }
    int sampleFormat;
    switch (this->metaData.bitsPerSample) {
        case 8: sampleFormat = SAMPLEFORMAT_UINT; break;
        case 16: sampleFormat = SAMPLEFORMAT_UINT; break;
        case 32: sampleFormat = SAMPLEFORMAT_IEEEFP; break;
        default: sampleFormat = SAMPLEFORMAT_UINT; break;
    }

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

    for (int i = 0; i < channel_vec[0].size(); ++i) {
        for (int c = 0; c < this->metaData.linChannels; ++c) {
            TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->metaData.imageWidth);
            TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->metaData.imageLength);
            TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, this->metaData.samplesPerPixel); // Jedes "Plane" ist ein separater Kanal
            TIFFSetField(out, TIFFTAG_BITSPERSAMPLE,this->metaData.bitsPerSample);
            TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
            TIFFSetField(out, TIFFTAG_PLANARCONFIG, this->metaData.planarConfig);
            TIFFSetField(out, TIFFTAG_PHOTOMETRIC, this->metaData.photometricInterpretation);
            if(!(this->metaData.description == "" || this->metaData.description.empty())){
                TIFFSetField(out, TIFFTAG_IMAGEDESCRIPTION, cutted_description.c_str());
            }
            TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, sampleFormat);

            for (int row = 0; row < channel_vec[c][i].rows; ++row) {
                if (TIFFWriteScanline(out, channel_vec[c][i].ptr(row), row, 0) < 0) { // '0' da jedes Plane ein Kanal ist
                    std::cerr << "[ERROR] Failed to write scanline for channel " << c << " at slice " << i << std::endl;
                    TIFFClose(out);
                    return false;
                }
            }
            TIFFWriteDirectory(out); // Starten eines neuen Directory für jeden Kanal pro Slice
        }
    }
    TIFFClose(out);

    return true;}
bool Hyperstack::saveAsTifDir(const std::string& directoryPath) const {
    // Create the directory if it does not exist
    if (!fs::exists(directoryPath)) {
        fs::create_directory(directoryPath);
    }

    std::vector<std::vector<cv::Mat>> channel_vec;
    for (auto& channel : this->channels) {
        channel_vec.push_back(channel.image.slices);
    }


    int c = 0;
    for (auto &channel: channel_vec) {
        cv::imwritemulti(directoryPath + "/channel_" + std::to_string(c) + ".tif", channel);
        c++;
    }
    return true;
}