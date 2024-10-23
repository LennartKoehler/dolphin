#include "HyperstackImage.h"
#include <tiffio.h>
#include <sstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "UtlIO.h"
#include "UtlFFT.h"
#include "PSF.h"
#include <filesystem>
#include <fftw3.h>

namespace fs = std::filesystem;

bool Hyperstack::readFromTifFile(const char *filename) {
    //METADATA
    std::string imageType = "";
    std::string name = "";
    std::string description = "";
    int imageWidth, imageLength = 0;
    int frameCount = 0;
    uint16_t resolutionUnit = 0;
    uint16_t samplesPerPixel = 0; //num of channels
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
    fileMetaData.name = name;
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
bool Hyperstack::readFromTifDir(const std::string& directoryPath) {
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "[ERROR] Specified path is not a directory or does not exist: " << directoryPath << std::endl;
        return false;
    }

    //METADATA
    std::string imageType = "";
    std::string name = directoryPath;
    const char* directoryName = directoryPath.c_str();
    std::string description = "";
    int imageWidth, imageLength = 0;
    int frameCount = 0;
    uint16_t resolutionUnit = 0;
    uint16_t samplesPerPixel = 0; //num of channels
    uint16_t bitsPerSample = 0;//bit depth
    uint16_t photometricInterpretation = 0;
    int linChannels = 1;//in Description (linearized channels)
    uint16_t planarConfig = 0;
    int totalImages = -1;
    int slices = 0;
    int dataType = 0; //calculated
    float xResolution, yResolution = 0.0f;
    std::vector<cv::Mat> layers;


    std::vector<std::string> fileNames;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.path().extension() == ".tif") {
            fileNames.push_back(entry.path().string());
        }
    }

    // Sort the files if necessary
    std::sort(fileNames.begin(), fileNames.end());

    // Load each image and add to the layer stack
    bool first_image = true;
    for (const auto& filename : fileNames) {
        cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if(first_image){
            TIFF* tifTempFile = TIFFOpen(filename.c_str(), "r");
            UtlIO::extractData(tifTempFile, name, description, directoryName, linChannels, slices, imageWidth, imageLength, resolutionUnit, xResolution, yResolution, samplesPerPixel, photometricInterpretation, bitsPerSample, frameCount, planarConfig, totalImages);
        }
        first_image = false;
        if (img.empty()) {
            std::cerr << "[ERROR] Failed to load image: " << filename << std::endl;
            continue;
        }

        layers.push_back(img);
        //TODO debug option or something
        //std::cout << "Loaded image: " << filename << std::endl;
    }

    if (layers.empty()) {
        std::cerr << "[ERROR] No images were loaded." << std::endl;
        return false;
    }

    // Set attributes based on the first image
    imageWidth = layers[0].cols;
    imageLength = layers[0].rows;
    // Depth based on the number of layers loaded
    totalImages = static_cast<int>(layers.size()) - 1;
    if(slices < 1){
        slices = static_cast<int>(layers.size());
    }
    //TODO debug
    std::cout<< "[INFO] Read in " << size(layers) << " layers"<< std::endl;

    //Converting Layers to 32F
    UtlIO::convertImageTo32F(layers, dataType, bitsPerSample);

    //Creating Channel Images
    if(linChannels > 0){
        std::vector<std::vector<cv::Mat>> channelData(linChannels);
        for(auto& channel : channelData){
            //cv::Mat data = cv::Mat(imageLength, imageWidth, dataType);
            //channel.push_back(data);
        }
        int c = 0;
        int z = 0;
        int multichannel_z = ((totalImages + 1) / linChannels);
        bool success = false;
        for(auto& singleLayer : layers){
            channelData[c].push_back(singleLayer);
            c++;
            if(c > linChannels-1){
                c = 0;
                z++;
            }
            //c = (c > this->getChannelNum()-1) ? (c = 0, ++sigmaZ) : c;
            if(multichannel_z == z) {
                std::cout<<"[INFO] " << name << " converted to multichannel" << std::endl;
                success = true;
            }
        }
        if(!success){
            std::cout << "[ERROR] "<< name << "(Layers: " << std::to_string(size(layers)) << ") could not converted to multichannel, Layers: " << std::to_string(z) << std::endl;
        }

        //create Image3D with channeldata
        int id = 0;
        for(auto& layers : channelData){
            Image3D imageLayers;
            imageLayers.slices = layers;
            Channel channel;
            channel.image = imageLayers;
            channel.id = id;
            this->channels.push_back(channel);
            id++;
        }

    }else{
        Image3D imageLayers;
        imageLayers.slices = layers;
        Channel channel;
        channel.image = imageLayers;
        channel.id = 0;
        this->channels.push_back(channel);
    }

    //Create Metadata
    ImageMetaData fileMetaData;
    fileMetaData.imageType = imageType;
    fileMetaData.name = name;
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
    fileMetaData.slices = slices;
    fileMetaData.dataType = dataType; //calculated
    fileMetaData.xResolution = xResolution;
    fileMetaData.yResolution = yResolution;

    this->metaData = fileMetaData;

    return true;
}

bool Hyperstack::saveAsTifFile(const std::string &directoryPath) {
    std::vector<std::vector<cv::Mat>> channel_vec;
    for (auto& channel : this->channels) {
        channel_vec.push_back(channel.image.slices);
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
bool Hyperstack::saveAsTifDir(const std::string& directoryPath) {
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