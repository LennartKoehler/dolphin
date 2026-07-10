#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin_image/ImagePadding.h"
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
    TestableReaderHandler(): ReaderHandler(std::unique_ptr<TiffReader>()){}
    BoxCoordWithPadding translateRegionRunner(const BoxCoordWithPadding& requestedRegion, const CuboidShape& imageSize) const {
        return translateRegion(requestedRegion, imageSize);
    }
};

class FileTestEnvironment : public ::testing::Environment {
public:
    ~FileTestEnvironment() override = default;
    void SetUp() override {}
    void TearDown() override { TestUtils::cleanupDirectory(); }
};


::testing::Environment* env = ::testing::AddGlobalTestEnvironment(new FileTestEnvironment());

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

    TiffReaderConfig readerConfig;
    readerConfig.numReaderThreads = 2;
    readerConfig.prefetchEnabled = true;
    readerConfig.prefetchCount = 4;
    ReaderHandler reader(std::make_unique<TiffReader>(path, 0, readerConfig));

    BoxCoordWithPadding box;
    box.box.position = CuboidPosition(0, 0, 0);
    box.box.dimensions = CuboidShape(2, 2, 2);
    box.padding.before = CuboidShape(0, 0, 0);
    box.padding.after = CuboidShape(0, 0, 0);
    PaddedImage readPadded = reader.getSubimage(box);
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
        TiffWriter writer(path, fullShape);
        for (size_t z = 0; z < fullShape.depth; z += chunkDepth) {
            BoxCoord subBox{CuboidPosition(0, 0, static_cast<int64_t>(z)),
                            CuboidShape(fullShape.width, fullShape.height, chunkDepth)};
            Image3D chunk = original.getSubimageCopy(subBox);

            BoxCoordWithPadding coord;
            coord.box.position = CuboidPosition(0, 0, static_cast<int64_t>(z));
            coord.box.dimensions = CuboidShape(fullShape.width, fullShape.height, chunkDepth);
            coord.padding.before = CuboidShape(0, 0, 0);
            coord.padding.after = CuboidShape(0, 0, 0);

            ASSERT_TRUE(writer.setSubimage(chunk, coord));
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
        TiffCompressionConfig config;
        config.compressionScheme = COMPRESSION_LZW;
        TiffWriter writer(path, fullShape, config);
        for (size_t z = 0; z < fullShape.depth; z += chunkDepth) {
            BoxCoord subBox{CuboidPosition(0, 0, static_cast<int64_t>(z)),
                            CuboidShape(fullShape.width, fullShape.height, chunkDepth)};
            Image3D chunk = original.getSubimageCopy(subBox);

            BoxCoordWithPadding coord;
            coord.box.position = CuboidPosition(0, 0, static_cast<int64_t>(z));
            coord.box.dimensions = CuboidShape(fullShape.width, fullShape.height, chunkDepth);
            coord.padding.before = CuboidShape(0, 0, 0);
            coord.padding.after = CuboidShape(0, 0, 0);

            ASSERT_TRUE(writer.setSubimage(chunk, coord));
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
        TiffWriter writer(path, fullShape);
        for (size_t zOff = 0; zOff < fullShape.depth; zOff += blockD) {
            for (size_t yOff = 0; yOff < fullShape.height; yOff += blockH) {
                for (size_t xOff = 0; xOff < fullShape.width; xOff += blockW) {
                    BoxCoord subBox{CuboidPosition(static_cast<int64_t>(xOff), static_cast<int64_t>(yOff), static_cast<int64_t>(zOff)),
                                    CuboidShape(blockW, blockH, blockD)};
                    Image3D block = original.getSubimageCopy(subBox);

                    BoxCoordWithPadding coord;
                    coord.box.position = CuboidPosition(static_cast<int64_t>(xOff), static_cast<int64_t>(yOff), static_cast<int64_t>(zOff));
                    coord.box.dimensions = CuboidShape(blockW, blockH, blockD);
                    coord.padding.before = CuboidShape(0, 0, 0);
                    coord.padding.after = CuboidShape(0, 0, 0);

                    ASSERT_TRUE(writer.setSubimage(block, coord));
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
        TiffWriter writer(path, fullShape);
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

            ASSERT_TRUE(writer.setSubimage(chunk, coord));
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), original.getShape());
    EXPECT_TRUE(readOpt->isEqual(original, 0.001f));
}
