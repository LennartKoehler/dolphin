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

#pragma once

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "Image3D.h"
#include "HyperstackImage.h"
#include "ImageMetaData.h"
#include "Channel.h"
#include <condition_variable>
#include <mutex>
#include <tiffio.h>

struct ImageBuffer{
    Image3D image;
    BoxCoordWithPadding source;
};

class ImageReader{
public:
    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const = 0;
};

class TiffReader : public ImageReader {
public:
    // Constructor
    // HyperstackReader();
    
    // Constructor with filename
    explicit TiffReader(std::string filename);
    
    // Destructor
    ~TiffReader();
    
    // Main methods
    bool readFromTifFile(std::string filename, std::vector<Channel>& channels, Image3D& layers);
    bool readFromTifDir(const std::string& directoryPath, std::vector<Channel>& channels, Image3D& layers);
    bool readFromFile(std::string filename, int y, int z, int height, int depth, Image3D& layers) const;
    PaddedImage getSubimage(const BoxCoordWithPadding& box) const override; // more explicitly mentioning what the padding is
    Image3D getSubimage(const BoxCoord& coord) const;
    Hyperstack getHyperstack(const BoxCoord& coord) const;
    
    
    // Getters
    const ImageMetaData& getMetaData() const;
    
private:
    mutable std::vector<ImageBuffer> loadedImageStrips; // since this class actually owns the data, perhaps keep track of if its still used or will be used in the future, each cube should be used only once
    ImageMetaData metaData;
    size_t maxBufferMemory_bytes;
    mutable size_t currentBufferMemory_bytes;

    mutable std::condition_variable memoryWaiter;
    mutable std::mutex mutex;
    
    // Helper methods

    void readStripWithPadding(const BoxCoordWithPadding& coord) const;
    int getBufferIndex(const BoxCoordWithPadding& coord) const;
    PaddedImage getFromBuffer(const BoxCoordWithPadding& coord, int bufferIndex) const;
    Image3D managedReader(int y, int z, int height, int depth) const;
    Image3D managedReader(const BoxCoord& coord) const;
    void updateCurrentMemoryBuffer(size_t memory) const;
    size_t getMemoryForShape(const RectangleShape& shape) const;
    void convertImageTo32F(Image3D& layers) const;
    void createImage3D(const Image3D& layers, std::vector<Channel>& channels);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static std::string getFilename(const std::string& path);
    
    // Separated metadata and file reading functions
    static bool extractMetadata(TIFF*& tifFile, ImageMetaData& metaData);
    static bool extractImageDataFromDirectory(const std::string& directoryPath, ImageMetaData& metaData);
    static bool readLayersFromTifFile(TIFF*& tifFile, Image3D& layers, const ImageMetaData& metaData);
    static bool readLayersFromDirectory(const std::string& directoryPath, Image3D& layers, ImageMetaData& metaData);
};