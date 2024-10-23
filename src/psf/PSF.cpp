#include <tiffio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "PSF.h"
#include "UtlIO.h"
namespace fs = std::filesystem;
#include "UtlImage.h"




bool PSF::readFromTifFile(const char *filename) {
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
    TIFF* tifOriginalFile = TIFFOpen(filename, "r");
    UtlIO::extractData(tifOriginalFile, name, description, filename, linChannels, slices, imageWidth, imageLength, resolutionUnit, xResolution, yResolution, samplesPerPixel, photometricInterpretation, bitsPerSample, frameCount, planarConfig, totalImages);

    //Read Layers
    UtlIO::readLayers(layers, totalImages, dataType, bitsPerSample, samplesPerPixel, tifOriginalFile);

    //Converting Layers to 32F
    UtlIO::convertImageTo32F(layers, dataType, bitsPerSample);
    UtlImage::normalizeToSumOne(layers);


    //Creating Channel Images
    Image3D imageLayers;
    std::vector<Channel> channels;
    UtlIO::createImage3D(channels, imageLayers, 0, totalImages, name, layers);
    this->image = imageLayers;

    return true;
}

bool PSF::readFromTifDir(const std::string &directoryPath) {
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "Specified path is not a directory or does not exist: " << directoryPath << std::endl;
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
            std::cerr << "Failed to load image: " << filename << std::endl;
            continue;
        }

        layers.push_back(img);
        //TODO debug option or something
        //std::cout << "Loaded image: " << filename << std::endl;
    }

    if (layers.empty()) {
        std::cerr << "No images were loaded." << std::endl;
        return false;
    }

    // Set attributes based on the first image
    imageWidth = layers[0].cols;
    imageLength = layers[0].rows;
    // Depth based on the number of layers loaded
    totalImages = static_cast<int>(layers.size()) - 1;

    //TODO debug
    std::cout<< "Read in " << size(layers) << " Slices"<< std::endl;

    //Converting Layers to 32F
    UtlIO::convertImageTo32F(layers, dataType, bitsPerSample);

    Image3D imageLayers;
    imageLayers.slices = layers;
    this->image = imageLayers;

    return true;
}

bool PSF::saveAsTifFile(const std::string &directoryPath) {
    std::vector<std::vector<cv::Mat>> channel_vec;
    channel_vec.push_back(this->image.slices);


    TIFF* out = TIFFOpen(directoryPath.c_str(), "w");
    if (!out) {
        std::cerr << "Cannot open TIFF file for writing: " << directoryPath.c_str() << std::endl;
        return false;
    }
    int sampleFormat;
    sampleFormat = SAMPLEFORMAT_IEEEFP;


    for (int i = 0; i < channel_vec[0].size(); ++i) {
            TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->image.slices[i].cols);
            TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->image.slices[i].rows);
            TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1); // Jedes "Plane" ist ein separater Kanal
            TIFFSetField(out, TIFFTAG_BITSPERSAMPLE,32);
            TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
            TIFFSetField(out, TIFFTAG_PLANARCONFIG, 1);
            TIFFSetField(out, TIFFTAG_PHOTOMETRIC, 1);
            TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, sampleFormat);

            for (int row = 0; row < channel_vec[0][i].rows; ++row) {
                if (TIFFWriteScanline(out, channel_vec[0][i].ptr(row), row, 0) < 0) { // '0' da jedes Plane ein Kanal ist
                    std::cerr << "Failed to write scanline for PSF at slice " << i << std::endl;
                    TIFFClose(out);
                    return false;
                }
            }
            TIFFWriteDirectory(out); // Starten eines neuen Directory für jeden Kanal pro Slice

    }
    TIFFClose(out);

    return true;}

bool PSF::saveAsTifDir(const std::string &directoryPath) {
        // Create the directory if it does not exist
    if (!fs::exists(directoryPath)) {
        fs::create_directory(directoryPath);
    }

    int i = 0;
    for(auto& slice : this->image.slices){
        // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
        cv::Mat img8bit;
        double minVal, maxVal;
        cv::minMaxLoc(slice, &minVal, &maxVal);

        // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
        // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
        slice.convertTo(img8bit, CV_8U, 255.0 / maxVal);
        std::string filename = directoryPath + "/Layer_" + std::to_string(i) + ".tif";
        cv::imwrite(filename, img8bit);
        i++;
    }

    return true;
}

// Flip the 3D PSF
PSF PSF::flip3DPSF() {
    PSF flippedPSF;
    int depth = this->image.slices.size();
    flippedPSF.image.slices.resize(depth); // Ensure the slices vector has the same size

    int z = 0;
    for (auto& layer : this->image.slices) {
        cv::Mat flippedSlice;
        cv::flip(layer, flippedSlice, -1);
        flippedPSF.image.slices[depth - 1 - z] = flippedSlice;
        z++;
    }

    return flippedPSF;
}
// Scaling the PSF
void PSF::scalePSF(int new_size_x, int new_size_y, int new_size_z) {
    int depth = this->image.slices.size();
    int height = this->image.slices[0].rows;
    int width = this->image.slices[0].cols;

    // Erstens: Skalieren Sie jede Ebene (2D-Bild) in der x- und y-Richtung
    std::vector<cv::Mat> scaled_psf_z;
    for (int i = 0; i < depth; ++i) {
        cv::Mat scaled_xy;
        cv::resize(this->image.slices[i], scaled_xy, cv::Size(new_size_x, new_size_y));
        scaled_psf_z.push_back(scaled_xy);
    }

    // Zweitens: Skalieren Sie entlang der Z-Achse
    std::vector<cv::Mat> final_scaled_psf;
    for (int i = 0; i < new_size_z; ++i) {
        int idx = static_cast<int>(i * (depth / static_cast<double>(new_size_z)));
        final_scaled_psf.push_back(scaled_psf_z[idx]);
    }

    // Ausgabe: Die skalierte PSF
    this->image.slices = final_scaled_psf;
}