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
#include <optional>
#include <future>
#include <vector>
#include <memory>
#include "dolphinbackend/MemoryComponent.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImageMetaData.h"
#include "dolphin_image/ImagePadding.h"
#include "dolphin_image/Types/PaddingFillType.h"

struct ReaderConfig {
    size_t numReaderThreads = 1;
    size_t readerMemory_byte = 0;
};


class ImageReader{
public:
    virtual ~ImageReader() = default;
    virtual void configure(int channel, ReaderConfig config = {}) = 0;
    virtual Image3D getSubimage(const BoxCoord& box) const = 0;
    virtual void prefetch(const std::vector<BoxCoord>& boxes) const {}
    virtual const ImageMetaData& getMetaData() const = 0;
    virtual size_t getRequiredMemory(const CuboidShape& subimageSize) const = 0;

    // MemoryComponent of this reader, for ONE region. Owned by the reader,
    // borrowed by the caller; the footprint applies the thread count.
    virtual MemoryComponent& getMemoryComponent(size_t nReaderThreads) = 0;

    static std::string getFilename(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }
        return path.substr(pos + 1);
    }

protected:
    std::unique_ptr<MemoryComponent> memoryComponent;
};


class ImageWriter {
public:
    virtual ~ImageWriter() = default;
    virtual bool setSubimage(const Image3D& image, const BoxCoord& coord,
                             const CuboidPosition& sourceOffset = CuboidPosition{0, 0, 0}) const = 0;
    // Write a host RealData (FFT-in-place real layout) directly into the
    // writer's buffers, no intermediate Image3D. Default: unsupported.
    virtual bool setSubimageReal(const RealData& data, const BoxCoord& coord,
                                 const CuboidPosition& sourceOffset = CuboidPosition{0, 0, 0}) const {
        (void)data; (void)coord; (void)sourceOffset;
        return false;
    }
    virtual void configure(WriterCompressionConfig compressionConfig, WriterConfig writerConfig = {}) = 0;

    // MemoryComponent of this writer. Owned by the writer, borrowed by the
    // caller; the footprint applies the count.
    virtual MemoryComponent& getMemoryComponent() = 0;

protected:
    std::unique_ptr<MemoryComponent> memoryComponent;
};



// Unpadded sub-cube plus the effective padding to apply (may be smaller than
// requested where the padded box is cropped at the image border).
struct UnpaddedCube {
    Image3D image;
    Padding padding;
};

// handles the padding offset and calls padding
class ReaderHandler{
public:
    explicit ReaderHandler(std::shared_ptr<ImageReader> r, PaddingFillType paddingFillStrategy) : reader(r), paddingFillStrategy(paddingFillStrategy){}
    const ImageMetaData& getMetaData() const { return reader->getMetaData(); }
    PaddingFillType getPaddingFillType() const { return paddingFillStrategy; }

    virtual PaddedImage getSubimage(const BoxCoordWithPadding& box) const{
        UnpaddedCube cube = getSubimageForPadding(box);
        ImagePadding::padImage(cube.image, cube.padding, paddingFillStrategy);
        return PaddedImage{std::move(cube.image), box.padding};
    }

    // Reads the sub-cube WITHOUT padding, together with the effective padding.
    // Callers that pad straight into their own buffer (e.g. the FFT-layout
    // host staging) use this to avoid the intermediate padded Image3D.
    virtual UnpaddedCube getSubimageForPadding(const BoxCoordWithPadding& box) const{
        BoxCoordWithPadding translatedRegion = translateRegion(box, reader->getMetaData().getShape());
        Image3D image = reader->getSubimage(translatedRegion.box);
        return UnpaddedCube{std::move(image), translatedRegion.padding};
    }

protected:
    BoxCoordWithPadding translateRegion(const BoxCoordWithPadding& requestedRegion, const CuboidShape& imageSize) const {
        BoxCoord paddedBox = requestedRegion.getPaddedBox();
        BoxCoord image = BoxCoord{CuboidShape{0,0,0}, imageSize};
        Padding padding = paddedBox.cropTo(image);
        return BoxCoordWithPadding{paddedBox, padding};
    }
    std::shared_ptr<ImageReader> reader;
    PaddingFillType paddingFillStrategy;
};


// strips padding from the image and forwards the real data to the underlying writer
class WriterHandler{
public:
    explicit WriterHandler(std::shared_ptr<ImageWriter> w) : writer(std::move(w)){}

    bool setSubimage(const Image3D& image, const BoxCoordWithPadding& coord) const {
        return writer->setSubimage(image, coord.box, coord.padding.before);
    }

    // Strips the padding and forwards the host RealData (FFT-in-place layout)
    // straight into the writer buffers, avoiding the intermediate Image3D.
    bool setSubimageReal(const RealData& data, const BoxCoordWithPadding& coord) const {
        return writer->setSubimageReal(data, coord.box, coord.padding.before);
    }

    void configure(WriterCompressionConfig compressionConfig, WriterConfig writerConfig = {}) {
        writer->configure(compressionConfig, writerConfig);
    }

private:
    std::shared_ptr<ImageWriter> writer;
};
