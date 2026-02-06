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
#include <itkImageRegionIterator.h>
#include "dolphin/IO/ReaderWriter.h"
#include "dolphin/IO/TiffExceptions.h"



class TiffReader : public ImageReader {
public:
    explicit TiffReader(const std::string& filename, int channel);
    
    ~TiffReader();
    
    static std::optional<Image3D> readTiffFile(const std::string& filename, int channel);
    static std::optional<ImageMetaData> readMetadata(const std::string& filename);
    
    // Non-static buffered reading methods
    std::optional<PaddedImage> getSubimage(const BoxCoordWithPadding& box) const override;
    const ImageMetaData& getMetaData() const override;
    

    
    
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
    

    static ImageMetaData readMetadata_(const std::string& filename);
    static void readSubimageFromTiffFile(TIFF* tiffile, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers, int channel);
    static void readSubimageFromTiffFileStatic(const std::string& filename, const ImageMetaData& metaData, int y, int z, int height, int depth, int width, Image3D& layers, int channel);
    static size_t getMemoryForShape(const CuboidShape& shape, const ImageMetaData& metaData);
    static void convertImageTo32F(Image3D& layers, const ImageMetaData& metaData);
    static ImageMetaData extractMetadataFromTiff(TIFF*& tifFile);
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static void convertScanlineToFloat(const char* scanlineData, std::vector<float>& rowData, int width, const ImageMetaData& metaData, int channel);

    static int countTiffDirectories(TIFF* tif);
};

