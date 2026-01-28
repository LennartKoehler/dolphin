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
#include <queue>

#include <tiffio.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include "dolphin/IO/ReaderWriter.h"


class TiffWriter : public ImageWriter {
public:
    // Constructor
    explicit TiffWriter(const std::string& filename, const RectangleShape& imageShape);
    
    // Destructor
    ~TiffWriter();
    
    // Main methods
    bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const override;
    
    bool saveToFile(const std::string& filename, int z, int depth, const Image3D& layers) const;
    
    // Static method for writing entire Image3D to file
    static bool writeToFile(const std::string& filename, const Image3D& image);
    
    // Getters
    // const ImageMetaData& getMetaData() const;
    
private:
    mutable std::mutex writerMutex;
    mutable CustomList<ImageBuffer> tileBuffer;
    // ImageMetaData metaData;
    std::string outputFilename;
    TIFF* tif;
    RectangleShape imageShape;
    mutable int writtenToDepth = 0;
    mutable std::queue<int> readyToWriteQueue; // Queue of tile indices ready to write
    // mutable std::map<int, bool> directories;
    
    // Helper methods
    static ImageMetaData extractMetaData(const Image3D& image);
    void createNewTile(const BoxCoordWithPadding& coord) const;
    bool isTileFull(const ImageBuffer& strip) const;
    bool writeTile(const std::string& filename, const ImageBuffer& strip) const;
    bool copyToTile(const Image3D& image, const BoxCoordWithPadding& coord, int index) const;
    void postprocessChannel(Image3D& image);
    int getStripIndex(const BoxCoordWithPadding& coord) const;
    void processReadyToWriteQueue() const;
    static void customTifWarningHandler(const char* module, const char* fmt, va_list ap);
    static bool writeSliceToTiff(TIFF* tif, const Image3D& image,  int sliceIndex);
    static void setTiffFields(TIFF* tif, const ImageMetaData& metaData);
    
    // Helper functions for ITK-based operations
    static int getTargetItkType(const ImageMetaData& metadata);
    static void extractSliceData(const Image3D& image, int sliceIndex, std::vector<float>& sliceData);
    static void convertSliceDataToTargetType(const std::vector<float>& sourceData, 
                                           std::vector<uint8_t>& targetData, 
                                           int width, int height,
                                           const ImageMetaData& metadata);

};