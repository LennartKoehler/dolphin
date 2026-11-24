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

#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin_image/ImagePadding.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "TestUtils.h"
#include <filesystem>
#include <tiffio.h>

class ReadWriteTest : public ::testing::Test {
protected:
    std::string testDir;

    void SetUp() override {
        testDir = TestUtils::outputPath();
    }

};

struct TestableReaderHandler : public ReaderHandler{
    TestableReaderHandler(): ReaderHandler(std::unique_ptr<TiffReader>(), PaddingFillType::MIRROR){}
    BoxCoordWithPadding translateRegionRunner(const BoxCoordWithPadding& requestedRegion, const CuboidShape& imageSize) const {
        return translateRegion(requestedRegion, imageSize);
    }
};

class FileTestEnvironment : public ::testing::Environment {
public:
    ~FileTestEnvironment() override = default;
    void SetUp() override { TestUtils::initLogging(); }
    void TearDown() override { TestUtils::cleanupDirectory(); }
};


inline ::testing::Environment* env = ::testing::AddGlobalTestEnvironment(new FileTestEnvironment());

// Metadata carrying only the shape; the writer normalizes the pixel format.
static ImageMetaData metaFromShape(const CuboidShape& shape) {
    ImageMetaData meta;
    meta.imageWidth = shape.width;
    meta.imageLength = shape.height;
    meta.slices = shape.depth;
    return meta;
}

TEST_F(ReadWriteTest, WriteAndReadRoundTrip) {
    Image3D original = TestUtils::createRandomImage(16, 16, 8);

    std::string path = testDir + "/rw_test.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, original));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());

    Image3D& read = *readOpt;
    EXPECT_EQ(read.getShape(), original.getShape());
    EXPECT_TRUE(read.isEqual(original, 0.001f));
}

TEST_F(ReadWriteTest, WriteConstantImage) {
    Image3D constant(CuboidShape(8, 8, 4), 3.5f);

    std::string path = testDir + "/rw_constant.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, constant));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), CuboidShape(8, 8, 4));
    for (auto it = readOpt->cbegin(); it != readOpt->cend(); ++it) {
        EXPECT_NEAR(*it, 3.5f, 0.001f);
    }
}

TEST_F(ReadWriteTest, ReadNonexistentFile) {
    auto result = TiffReader::readTiffFile("/nonexistent/file.tif", 0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(ReadWriteTest, ReadMetadata) {
    Image3D img(CuboidShape(12, 10, 6), 1.0f);
    std::string path = testDir + "/rw_metadata.tif";
    TiffWriter::writeToFile(path, img);

    auto metaOpt = TiffReader::readMetadata(path);
    ASSERT_TRUE(metaOpt.has_value());
    EXPECT_EQ(metaOpt->imageWidth, 12);
    EXPECT_EQ(metaOpt->imageLength, 10);
    EXPECT_EQ(metaOpt->slices, 6);
}

TEST_F(ReadWriteTest, WriteImpulseImage) {
    Image3D impulse = TestUtils::createImpulseImage(8, 8, 8);

    std::string path = testDir + "/rw_impulse.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, impulse));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_TRUE(readOpt->isEqual(impulse, 0.001f));
}


TEST_F(ReadWriteTest, ReadSubimage) {
    Image3D gradient = TestUtils::createGradientImage(8, 8, 8);

    std::string path = testDir + "/rw_subimage_random.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, gradient));

    ReaderConfig readerConfig;
    readerConfig.numReaderThreads = 2;
    auto tr = std::make_unique<TiffReader>(path);
    tr->configure(0, readerConfig);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);

    BoxCoordWithPadding box;
    box.box.position = CuboidPosition(0, 0, 0);
    box.box.dimensions = CuboidShape(2, 2, 2);
    box.padding.before = CuboidShape(0, 0, 0);
    box.padding.after = CuboidShape(0, 0, 0);
    PaddedImage readPadded = reader.getSubimagePadded(box);
    Image3D expectedResult = TestUtils::createGradientImage(2, 2, 2);

    EXPECT_EQ(readPadded.image.getShape(), expectedResult.getShape());
    EXPECT_TRUE(readPadded.image.isEqual(expectedResult, 0.001f));
}


TEST_F(ReadWriteTest, PaddingConversionFullWithin) {
    TestableReaderHandler handler{};
    BoxCoord region{CuboidPosition(5,5,5), CuboidShape(8,8,8)};
    Padding padding{CuboidShape(2,2,2), CuboidShape(2,2,2)};
    BoxCoordWithPadding requestedRegion{region, padding};
    CuboidShape imageSize{20, 20, 20};

    BoxCoordWithPadding result = handler.translateRegionRunner(requestedRegion, imageSize);

    BoxCoord expectedBox{CuboidPosition(3,3,3),CuboidShape(12,12,12)};
    Padding expectedPadding{CuboidShape(0,0,0),CuboidShape(0,0,0)};
    EXPECT_EQ(result.box, expectedBox);
    EXPECT_EQ(result.padding, expectedPadding);
}


TEST_F(ReadWriteTest, PaddingConversionNotWithin) {
    TestableReaderHandler handler{};
    BoxCoord region{CuboidPosition(1,1,1), CuboidShape(8,8,8)};
    Padding padding{CuboidShape(2,2,2), CuboidShape(2,2,2)};
    BoxCoordWithPadding requestedRegion{region, padding};
    CuboidShape imageSize{10, 10, 10};

    BoxCoordWithPadding result = handler.translateRegionRunner(requestedRegion, imageSize);

    BoxCoord expectedBox{CuboidPosition(0,0,0),CuboidShape(10,10,10)};
    Padding expectedPadding{CuboidShape(1,1,1),CuboidShape(1,1,1)};
    EXPECT_TRUE(result.box == expectedBox);
    EXPECT_TRUE(result.padding == expectedPadding);
}


TEST_F(ReadWriteTest, SetSubimageDepthSplit) {
    Image3D original = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/rw_setsubimage_depth.tif";

    CuboidShape fullShape(16, 16, 8);
    size_t chunkDepth = 2;

    {
        TiffWriter writer(path, TiffWriter::extractMetaData(original));
        for (size_t z = 0; z < fullShape.depth; z += chunkDepth) {
            BoxCoord subBox{CuboidPosition(0, 0, static_cast<int64_t>(z)),
                            CuboidShape(fullShape.width, fullShape.height, chunkDepth)};
            Image3D chunk = original.getSubimageCopy(subBox);

            ASSERT_TRUE(writer.setSubimage(chunk, subBox));
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), original.getShape());
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}


TEST_F(ReadWriteTest, SetSubimageWithLZWCompression) {
    Image3D original = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/rw_setsubimage_lzw.tif";

    CuboidShape fullShape(16, 16, 8);
    size_t chunkDepth = 2;

    {
        WriterCompressionConfig config;
        config.compressionScheme = COMPRESSION_LZW;
        TiffWriter writer(path, TiffWriter::extractMetaData(original));
        writer.configure(config);  // WriterConfig defaults to strips
        for (size_t z = 0; z < fullShape.depth; z += chunkDepth) {
            BoxCoord subBox{CuboidPosition(0, 0, static_cast<int64_t>(z)),
                            CuboidShape(fullShape.width, fullShape.height, chunkDepth)};
            Image3D chunk = original.getSubimageCopy(subBox);

            ASSERT_TRUE(writer.setSubimage(chunk, subBox));
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), original.getShape());
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}


TEST_F(ReadWriteTest, SetSubimageSpatialSplit) {
    Image3D original = TestUtils::createGradientImage(16, 16, 4);
    std::string path = testDir + "/rw_setsubimage_spatial.tif";

    CuboidShape fullShape(16, 16, 4);
    size_t blockW = 8, blockH = 8, blockD = 4;

    {
        TiffWriter writer(path, TiffWriter::extractMetaData(original));
        for (size_t zOff = 0; zOff < fullShape.depth; zOff += blockD) {
            for (size_t yOff = 0; yOff < fullShape.height; yOff += blockH) {
                for (size_t xOff = 0; xOff < fullShape.width; xOff += blockW) {
                    BoxCoord subBox{CuboidPosition(static_cast<int64_t>(xOff), static_cast<int64_t>(yOff), static_cast<int64_t>(zOff)),
                                    CuboidShape(blockW, blockH, blockD)};
                    Image3D block = original.getSubimageCopy(subBox);

                    ASSERT_TRUE(writer.setSubimage(block, subBox));
                }
            }
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), original.getShape());
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}


TEST_F(ReadWriteTest, SetSubimageWithPadding) {
    Image3D original = TestUtils::createGradientImage(8, 8, 4);
    std::string path = testDir + "/rw_setsubimage_padding.tif";

    CuboidShape fullShape(8, 8, 4);
    size_t chunkDepth = 2;
    Padding pad{CuboidShape(2, 2, 2), CuboidShape(2, 2, 2)};

    {
        auto tiffWriter = std::make_shared<TiffWriter>(path, TiffWriter::extractMetaData(original));
        WriterHandler writerHandler(tiffWriter);
        for (size_t z = 0; z < fullShape.depth; z += chunkDepth) {
            BoxCoord subBox{CuboidPosition(0, 0, static_cast<int64_t>(z)),
                            CuboidShape(fullShape.width, fullShape.height, chunkDepth)};
            Image3D chunk = original.getSubimageCopy(subBox);

            ImagePadding::padImageMirror(chunk, pad);

            BoxCoordWithPadding coord;
            coord.box.position = CuboidPosition(0, 0, static_cast<int64_t>(z));
            coord.box.dimensions = CuboidShape(fullShape.width, fullShape.height, chunkDepth);
            coord.padding.before = pad.before;
            coord.padding.after = pad.after;

            ASSERT_TRUE(writerHandler.setSubimage(chunk, coord));
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), original.getShape());
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}

// --- Source metadata carry-over ------------------------------------------------
// The writer receives the reader's metadata and must write the descriptive
// tags back verbatim (ImageDescription, X/YResolution, ResolutionUnit), while
// the pixel format stays writer-controlled (float32).

// Creates a minimal single-directory float32 TIFF carrying an ImageDescription
// and resolution tags, the way an external microscopy tool would.
static void writeMetadataSourceTiff(const std::string& path) {
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    ASSERT_NE(tif, nullptr);
    const uint32_t width = 4, length = 4;
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, length);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, length);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, "channels=1\nslices=3");
    TIFFSetField(tif, TIFFTAG_XRESOLUTION, 300.0f);
    TIFFSetField(tif, TIFFTAG_YRESOLUTION, 200.0f);
    TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
    std::vector<float> strip(width * length, 0.5f);
    ASSERT_EQ(TIFFWriteEncodedStrip(tif, 0, strip.data(), strip.size() * sizeof(float)), strip.size() * sizeof(float));
    ASSERT_TRUE(TIFFWriteDirectory(tif));
    TIFFClose(tif);
}

TEST_F(ReadWriteTest, WriterCarriesOverSourceMetadata) {
    std::string srcPath = testDir + "/meta_source.tif";
    writeMetadataSourceTiff(srcPath);

    auto srcMetaOpt = TiffReader::readMetadata(srcPath);
    ASSERT_TRUE(srcMetaOpt.has_value());
    ImageMetaData srcMeta = *srcMetaOpt;
    EXPECT_EQ(srcMeta.description, "channels=1\nslices=3");
    EXPECT_EQ(srcMeta.linChannels, 1);
    EXPECT_EQ(srcMeta.slices, 3u);
    EXPECT_FLOAT_EQ(srcMeta.xResolution, 300.0f);
    EXPECT_FLOAT_EQ(srcMeta.yResolution, 200.0f);
    EXPECT_EQ(srcMeta.resolutionUnit, RESUNIT_CENTIMETER);

    std::string outPath = testDir + "/meta_output.tif";
    Image3D image = TestUtils::createGradientImage(4, 4, 3);
    {
        TiffWriter writer(outPath, srcMeta);
        BoxCoord full{CuboidPosition(0, 0, 0), CuboidShape(4, 4, 3)};
        ASSERT_TRUE(writer.setSubimage(image, full));
    }

    auto outMetaOpt = TiffReader::readMetadata(outPath);
    ASSERT_TRUE(outMetaOpt.has_value());
    const ImageMetaData& outMeta = *outMetaOpt;
    EXPECT_EQ(outMeta.description, srcMeta.description);
    EXPECT_EQ(outMeta.linChannels, srcMeta.linChannels);
    EXPECT_EQ(outMeta.slices, srcMeta.slices);
    EXPECT_FLOAT_EQ(outMeta.xResolution, srcMeta.xResolution);
    EXPECT_FLOAT_EQ(outMeta.yResolution, srcMeta.yResolution);
    EXPECT_EQ(outMeta.resolutionUnit, srcMeta.resolutionUnit);

    // Pixel data path unchanged: float32, correct shape and values.
    EXPECT_EQ(outMeta.bitsPerSample, 32);
    EXPECT_EQ(outMeta.sampleFormat, SAMPLEFORMAT_IEEEFP);
    auto readOpt = TiffReader::readTiffFile(outPath, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), image.getShape());
    EXPECT_TRUE(readOpt->isEqual(image, 0.001f));
}

// --- TiffWriterMemoryComponent: reports the writer's PEAK held at once --------

TEST_F(ReadWriteTest, WriterMemoryComponentTwoTilesPlusWriteBuffers) {
    const CuboidShape shape(16, 8, 4);
    std::string path = testDir + "/rw_mem_component.tif";

    TiffWriter writer(path, metaFromShape(shape));
    MemoryComponent& component = writer.getMemoryComponent();
    EXPECT_EQ(component.name(), "tiffwriter");

    // At most two tiles buffered at once, each full width/height x one write
    // depth, plus the transient write path (one strip buffer per written
    // slice + libtiff's internal encode buffer; stripped mode = one full
    // slice each).
    const CuboidShape writeShape(4, 4, 2);
    const size_t tileBytes = shape.width * shape.height * writeShape.depth * sizeof(float);
    const size_t sliceBytes = shape.width * shape.height * sizeof(float);
    const MemoryRequirement need = component.footprint(writeShape);
    EXPECT_EQ(need.hostBytes, 2 * tileBytes + 2 * sliceBytes);
    EXPECT_EQ(need.deviceBytes, 0u);
}

TEST_F(ReadWriteTest, WriterMemoryComponentScalesWithWriteDepth) {
    const CuboidShape shape(16, 8, 4);
    std::string path = testDir + "/rw_mem_component_shape.tif";

    TiffWriter writer(path, metaFromShape(shape));
    MemoryComponent& component = writer.getMemoryComponent();

    // The tile backlog depends on the candidate write depth: deeper writes
    // buffer more, equal-depth candidates report the same demand.
    const size_t shallow = component.footprint(CuboidShape(4, 4, 2)).hostBytes;
    const size_t deep = component.footprint(CuboidShape(16, 8, 4)).hostBytes;
    EXPECT_GT(deep, shallow);
    EXPECT_EQ(component.footprint(CuboidShape(64, 64, 2)).hostBytes, shallow);
}

TEST_F(ReadWriteTest, WriterMemoryComponentTiledWriteBuffer) {
    const CuboidShape shape(16, 8, 4);
    std::string path = testDir + "/rw_mem_component_tiled.tif";

    TiffWriter writer(path, metaFromShape(shape));
    WriterCompressionConfig compression;
    WriterConfig config;
    config.tileWidth = 4;
    config.tileLength = 4;
    writer.configure(compression, config);

    // Tiled mode: the transient write path buffers one tile per written
    // slice instead of one full slice; the two-tile backlog term is
    // unchanged.
    const CuboidShape writeShape(4, 4, 2);
    const size_t tileBytes = shape.width * shape.height * writeShape.depth * sizeof(float);
    const size_t writeBufferBytes = 4 * 4 * sizeof(float);
    const MemoryRequirement need = writer.getMemoryComponent().footprint(writeShape);
    EXPECT_EQ(need.hostBytes, 2 * tileBytes + 2 * writeBufferBytes);
}

TEST_F(ReadWriteTest, WriterMemoryComponentIsStableInstance) {
    std::string path = testDir + "/rw_mem_component_instance.tif";
    TiffWriter writer(path, metaFromShape(CuboidShape(8, 8, 4)));

    MemoryComponent& first = writer.getMemoryComponent();
    MemoryComponent& second = writer.getMemoryComponent();
    EXPECT_EQ(&first, &second);
}

// Pads `interior` (at the origin) into a FFT-in-place host RealData with
// `padding`, exactly as the deconvolution executor stages a cube.
static RealData chunkToReal(const Image3D& interior, const CuboidShape& interiorShape, const Padding& padding) {
    const CuboidShape workShape = interiorShape + padding.before + padding.after;
    RealData out = BackendFactory::getInstance().getHostBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);
    Preprocessor::convertImageToRealDataPadded(Image3D(interior), padding, PaddingFillType::ZERO, out);
    return out;
}

// setSubimageReal writes the host RealData (FFT-in-place layout, x-strided)
// straight into the writer. Covering the whole image in z-chunks, the result
// must be identical to writing each chunk through the Image3D path (validates
// the strided FFT-row copy).
TEST_F(ReadWriteTest, SetSubimageRealFullWidthRoundTrip) {
    const CuboidShape fullShape(16, 16, 8);
    const Image3D original = TestUtils::createGradientImage(16, 16, 8);
    const size_t chunkDepth = 4;
    const Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    std::string path = testDir + "/rw_setsubimage_real.tif";

    {
        WriterHandler handler(std::make_shared<TiffWriter>(path, TiffWriter::extractMetaData(original)));
        for (size_t z = 0; z < fullShape.depth; z += chunkDepth) {
            BoxCoord subBox{CuboidPosition(0, 0, static_cast<int64_t>(z)),
                            CuboidShape(fullShape.width, fullShape.height, chunkDepth)};
            Image3D interior = original.getSubimageCopy(subBox);
            BoxCoordWithPadding coord;
            coord.box = subBox;
            coord.padding = padding;
            RealData real = chunkToReal(interior, subBox.dimensions, padding);
            ASSERT_TRUE(handler.setSubimageReal(real, coord));
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), fullShape);
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}

// Spatial (partial width/height/depth) blocks with user padding, covering the
// whole image: exercises the strided x copy AND the sourceOffset strip (the
// padding is stripped before writing).
TEST_F(ReadWriteTest, SetSubimageRealSpatialCubeRoundTrip) {
    const CuboidShape fullShape(16, 16, 8);
    const Image3D original = TestUtils::createGradientImage(16, 16, 8);
    const size_t blockW = 8, blockH = 8, blockD = 4;
    const Padding padding{CuboidShape(1, 1, 1), CuboidShape(1, 1, 1)};
    std::string path = testDir + "/rw_setsubimage_real_spatial.tif";

    {
        WriterHandler handler(std::make_shared<TiffWriter>(path, TiffWriter::extractMetaData(original)));
        for (size_t zOff = 0; zOff < fullShape.depth; zOff += blockD) {
            for (size_t yOff = 0; yOff < fullShape.height; yOff += blockH) {
                for (size_t xOff = 0; xOff < fullShape.width; xOff += blockW) {
                    BoxCoord subBox{CuboidPosition(static_cast<int64_t>(xOff), static_cast<int64_t>(yOff), static_cast<int64_t>(zOff)),
                                    CuboidShape(blockW, blockH, blockD)};
                    Image3D interior = original.getSubimageCopy(subBox);
                    BoxCoordWithPadding coord;
                    coord.box = subBox;
                    coord.padding = padding;
                    RealData real = chunkToReal(interior, subBox.dimensions, padding);
                    ASSERT_TRUE(handler.setSubimageReal(real, coord));
                }
            }
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), fullShape);
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}
