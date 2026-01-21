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
#include "Image3D.h"
#include "HyperstackImage.h"
#include "ImageMetaData.h"
#include "Channel.h"
#include <condition_variable>
#include <mutex>
#include <tiffio.h>
#include "itkImageRegionIterator.h"


class ImageReader{
public:
    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const = 0;
    virtual const ImageMetaData& getMetaData() const = 0;
    static std::string getFilename(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path; // No directory separator found, return whole string
        }
        return path.substr(pos + 1);
    }
};

class TiffReader : public ImageReader {
public:
    // Constructor with filename - now only for buffered reading
    explicit TiffReader(std::string filename, int channel);
    
    // Destructor
    ~TiffReader();
    
    // Main static methods    static Image3D readTiffFile(const std::string& filename, int channel);
    static Image3D readTiffFile(const std::string& filename, int channel);
    static ImageMetaData extractMetadata(const std::string& filename);
    static bool readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers, int channel);
    
    // Non-static version that uses member TIFF* variable
    bool readSubimageFromTiffFile(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers) const ;
    
    // Non-static buffered reading methods
    PaddedImage getSubimage(const BoxCoordWithPadding& box) const override;
    const ImageMetaData& getMetaData() const override;
    
    // Legacy methods (deprecated)
    bool readFromTifDir(const std::string& directoryPath, std::vector<Channel>& channels, Image3D& layers);
    
    // Getters
    
    
private:
    // Buffer management for getSubimage
    mutable CustomList<ImageBuffer> loadedImageStrips;
    mutable ImageMetaData metaData;
    size_t maxBufferMemory_bytes;
    mutable size_t currentBufferMemory_bytes;
    mutable std::condition_variable memoryWaiter;
    mutable std::mutex mutex;
    int channel;
    TIFF* tif; // Member variable to keep TIFF file open
    
    // Non-static helper methods for buffered reading
    void readStripWithPadding(const BoxCoordWithPadding& coord) const;
    int getStripIndex(const BoxCoordWithPadding& coord) const;
    PaddedImage getFromBuffer(const BoxCoordWithPadding& coord, int bufferIndex) const;
    void updateCurrentMemoryBuffer(size_t memory) const;
    Image3D managedReader(const BoxCoord& coord) const;
    void createImage3D(const Image3D& layers, std::vector<Channel>& channels);
    
    // Static helper methods
    static size_t getMemoryForShape(const RectangleShape& shape, const ImageMetaData& metaData);
    static void convertImageTo32F(Image3D& layers, const ImageMetaData& metaData);
    static ImageMetaData extractMetadataFromTiff(TIFF*& tifFile);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, int width, const ImageMetaData& metaData);

    static int countTiffDirectories(TIFF* tif);
};