#pragma once

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "Image3D.h"
#include "HyperstackImage.h"

namespace UtlIO{
    void convertImageTo32F(std::vector<cv::Mat> &layers, int &dataType, uint16_t &bitsPerSample);

    bool readLayers(std::vector<cv::Mat> &layers, int &totalimages, int &dataType, uint16_t &bitsPerSample, uint16_t &samplesPerPixel, TIFF* &tifOriginalFile);

    bool extractData(TIFF* &tifOriginalFile, std::string &name, std::string &description, const char* &filename, int &linChannels, int &slices, int &imageWidth, int &imageLength, uint16_t &resolutionUnit, float &xResolution, float &yResolution, uint16_t &samplesPerPixel, uint16_t &photometricInterpretation, uint16_t &bitsPerSample, int &frameCount, uint16_t &planarConfig, int &totalimages);

    void createImage3D(std::vector<Channel> &channels, Image3D &imageLayers, int linChannels, int totalImages, std::string name, std::vector<cv::Mat> layers);

    void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
}

