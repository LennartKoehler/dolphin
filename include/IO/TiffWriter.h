#pragma once

#include <string>
#include <vector>
#include <queue>
#include <opencv2/core/mat.hpp>
#include "Image3D.h"
#include "HyperstackImage.h"
#include "ImageMetaData.h"
#include "Channel.h"
#include <tiffio.h>
class ImageWriter {
public:
    virtual bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const = 0;
};

class TiffWriter : public ImageWriter {
public:
    // Constructor
    explicit TiffWriter(const std::string& filename, const ImageMetaData& metadata);
    
    // Destructor
    ~TiffWriter();
    
    // Main methods
    bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const override;
    
    bool saveToFile(const std::string& filename, int y, int z, int height, int depth, const Image3D& layers) const;
    
    // Static method for writing entire Image3D to file
    static bool writeToFile(const std::string& filename, const Image3D& image, const ImageMetaData& metadata);
    
    // Getters
    const ImageMetaData& getMetaData() const;
    
private:
    mutable std::mutex writerMutex;
    ImageMetaData metaData;
    mutable CustomList<ImageBuffer> tileBuffer;
    TIFF* tif;
    mutable int writtenToDepth = 0;
    mutable std::queue<int> readyToWriteQueue; // Queue of tile indices ready to write
    // mutable std::map<int, bool> directories;
    
    // Helper methods
    void createNewTile(const BoxCoordWithPadding& coord) const;
    bool isTileFull(const ImageBuffer& strip) const;
    bool writeTile(const std::string& filename, const ImageBuffer& strip) const;
    bool copyToTile(const Image3D& image, const BoxCoordWithPadding& coord, int index) const;
    void postprocessChannel(Image3D& image);
    int getStripIndex(const BoxCoordWithPadding& coord) const;
    void processReadyToWriteQueue() const;
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    bool writeImageToTiff(const cv::Mat& image, int directoryIndex, int yOffset) const;

};