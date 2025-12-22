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

#include <iostream>
#include <tiffio.h>
#include <sstream>
#include "UtlIO.h"
#include "UtlImage.h"
#include "HyperstackImage.h"
#include <vector>
#include <opencv2/opencv.hpp>

void UtlIO::customTifWarningHandler(const char* module, const char* fmt, va_list ap) {
    // Ignoriere alle Warnungen oder filtere nach bestimmten Tags
    // Beispiel: printf(fmt, ap); // Um die Warnungen anzuzeigen
}

void UtlIO::convertImageTo32F(std::vector<cv::Mat> &layers, int &dataType, uint16_t &bitsPerSample){
    double globalMin, globalMax;
    UtlImage::findGlobalMinMax(layers, globalMin, globalMax);

    int i = 0;
    for (auto& layer : layers) {
        int type = CV_MAKETYPE(CV_32F, layer.channels());
        layer.convertTo(layer, type, 1 / (globalMax - globalMin), -globalMin * (1 / (globalMax - globalMin)));

        std::cout << "\r[STATUS] Layer " << i << "/" << layers.size() - 1 << " in 32F converted"
                  << " ";
        std::flush(std::cout);
        i++;
    }
    std::cout << std::to_string(layers[1].at<float>(1, 1)) << std::endl;

    bitsPerSample = 32;
    dataType = CV_32F;
}

bool UtlIO::readLayers(std::vector<cv::Mat> &layers, int &totalImages, int &dataType, uint16_t &bitsPerSample, uint16_t &samplesPerPixel, TIFF* &tifOriginalFile){
    if (bitsPerSample == 8) {
        dataType = CV_8UC(samplesPerPixel);
    } else if (bitsPerSample == 16) {
        dataType = CV_16UC(samplesPerPixel);
    } else if (bitsPerSample == 32) {
        dataType = CV_32FC(samplesPerPixel);
    } else {
        std::cerr << bitsPerSample << "[ERROR] Unsupported bit depth." << std::endl;
        dataType = 0;
        return false;
    }
    do {
        totalImages++;
        uint32_t width, height;
        TIFFGetField(tifOriginalFile, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tifOriginalFile, TIFFTAG_IMAGELENGTH, &height);
        tsize_t scanlineSize = TIFFScanlineSize(tifOriginalFile);

        cv::Mat layer = cv::Mat(height, width, dataType);

        char* buf;
        uint32_t row;
        buf = (char *)_TIFFmalloc(scanlineSize);
        if (!buf) {
            std::cerr << "[ERROR] Memory allocation failed for buffer." << std::endl;
            TIFFClose(tifOriginalFile);
            return false;
        }

        for (row = 0; row < height; row++) {
            TIFFReadScanline(tifOriginalFile, buf, row);
            memcpy(layer.ptr(row), buf, scanlineSize);
        }

        //layer.channelData = channelData;
        layers.push_back(layer);

        _TIFFfree(buf);

        //TODO debug option
        //std::cout << "Layer  " << depth << " successfully read" << std::endl;
    } while (TIFFReadDirectory(tifOriginalFile));

    //TODO debug
    std::cout<< "[INFO] Read in " << layers.size() << " layers"<< std::endl;


    return true;
}

bool UtlIO::extractData(TIFF* &tifOriginalFile, std::string &name, std::string &description, const char* &filename, int &linChannels, int &slices, int &imageWidth, int &imageLength, uint16_t &resolutionUnit, float &xResolution, float &yResolution, uint16_t &samplesPerPixel, uint16_t &photometricInterpretation, uint16_t &bitsPerSample, int &frameCount, uint16_t &planarConfig, int &totalimages){
    if (!tifOriginalFile) {
        std::cerr << "[ERROR] Cannot open TIFF file: " << filename << std::endl;
        return false;
    }else{
        //Excract Data
        name = filename;
        char* img_description;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_IMAGEDESCRIPTION, &img_description)) {
            std::string desc(img_description);
            std::istringstream iss(desc);
            std::string line;
            while (getline(iss, line)) {
                if (line.find("channels=") != std::string::npos) {
                    linChannels = std::stoi(line.substr(line.find("=") + 1));
                }
                if (line.find("slices=") != std::string::npos) {
                    slices = std::stoi(line.substr(line.find("=") + 1));
                }
            }
            description = desc;
        } else {
            std::cout << "[INFO] No image description" << std::endl;
        }

        // If slices not found in description, calculate from total images and channels
        if (slices == 0) {
            int totalDirectories = countTiffDirectories(tifOriginalFile);
            
            if (linChannels > 0) {
                slices = totalDirectories / linChannels;
                std::cout << "[INFO] Calculated slices from total directories: " << slices 
                         << " (total dirs: " << totalDirectories << ", channels: " << linChannels << ")" << std::endl;
            } else {
                slices = totalDirectories;
                std::cout << "[INFO] Using total directories as slices: " << slices << std::endl;
            }
        }

        // Lese TIFF-Tags, wie im früheren Beispiel
        int tempWidth, tempLength, tempResUnit, tempspp, temppi, tempbps, tempfc, temppc;
        float tempXRes, tempYRes;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_IMAGEWIDTH, &tempWidth))
            imageWidth = tempWidth;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_IMAGELENGTH, &tempLength))
            imageLength = tempLength;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_RESOLUTIONUNIT, &tempResUnit))
            resolutionUnit = tempResUnit;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_XRESOLUTION, &tempXRes) && tempXRes > 0)
            xResolution = tempXRes;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_YRESOLUTION, &tempYRes) && tempYRes > 0)
            yResolution = tempYRes;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_SAMPLESPERPIXEL, &tempspp) && tempspp > 0)
            samplesPerPixel = tempspp;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_PHOTOMETRIC, &temppi))
            photometricInterpretation = temppi;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_BITSPERSAMPLE, &tempbps))
            bitsPerSample = tempbps;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_FRAMECOUNT, &tempfc))
            frameCount = tempfc;
        if (TIFFGetField(tifOriginalFile, TIFFTAG_PLANARCONFIG, &temppc)) {
            planarConfig = temppc;
        }

        return true;

    }
}

void UtlIO::createImage3D(std::vector<Channel> &channels, Image3D &imageLayers, int linChannels, int totalImages, std::string name, std::vector<cv::Mat> layers){
    if(linChannels > 0){
        std::vector<std::vector<cv::Mat>> channelData(linChannels);
        int c = 0;
        int z = 0;
        int multichannel_z = ((totalImages + 1) / linChannels);
        
        // Calculate slices per channel
        int slicesPerChannel = layers.size() / linChannels;
        std::cout << "[INFO] Calculated " << slicesPerChannel << " slices per channel" << std::endl;
        
        bool success = false;
        for(auto& singleLayer : layers){
            channelData[c].push_back(singleLayer);
            c++;
            if(c > linChannels-1){
                c = 0;
                z++;
            }
            if(multichannel_z == z) {
                std::cout <<"[INFO] " << name << " converted to multichannel" << std::endl;
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
            channels.push_back(channel);
            id++;
        }

    }else{
        imageLayers.slices = layers;
        Channel channel;
        channel.image = imageLayers;
        channel.id = 0;
        channels.push_back(channel);
    }
}



std::string UtlIO::getFilename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path; // No directory separator found, return whole string
    }
    return path.substr(pos + 1);
}

int UtlIO::countTiffDirectories(TIFF* tif) {
    int count = 0;
    do {
        count++;
    } while (TIFFReadDirectory(tif));
    
    // Reset to first directory
    TIFFSetDirectory(tif, 0);
    return count;
}