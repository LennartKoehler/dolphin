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

#include <condition_variable>
#include <mutex>
#include <tiffio.h>
#include "itkImageRegionIterator.h"
#include "ReaderWriter.h"



class TiffReader : public ImageReader {
public:
    // Constructor with filename - now only for buffered reading
    explicit TiffReader(const std::string& filename, int channel);
    
    // Destructor
    ~TiffReader();
    
    // Main static methods    static Image3D readTiffFile(const std::string& filename, int channel);
    static Image3D readTiffFile(const std::string& filename, int channel);
    static ImageMetaData extractMetadataStatic(const std::string& filename);
    static bool readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers, int channel);
    
    // Non-static version that uses member TIFF* variable
    static bool readSubimageFromTiffFile(TIFF* tiffile, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers, int channel);
    
    ImageMetaData extractMetadata();
    // Non-static buffered reading methods
    PaddedImage getSubimage(const BoxCoordWithPadding& box) const override;
    const ImageMetaData& getMetaData() const override;
    
    // Legacy methods (deprecated)
    
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
    
    // Static helper methods
    static size_t getMemoryForShape(const RectangleShape& shape, const ImageMetaData& metaData);
    static void convertImageTo32F(Image3D& layers, const ImageMetaData& metaData);
    static ImageMetaData extractMetadataFromTiff(TIFF*& tifFile);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, int width, const ImageMetaData& metaData, int channel);

    static int countTiffDirectories(TIFF* tif);
};

