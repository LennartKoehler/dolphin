#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "TestUtils.h"
#include <future>
#include <vector>

class LoggingEnvironment : public ::testing::Environment {
public:
    void SetUp() override { TestUtils::initLogging(); }
};
inline ::testing::Environment* logEnv = ::testing::AddGlobalTestEnvironment(new LoggingEnvironment());

namespace {

void expectImageEquals(const Image3D& actual, const Image3D& expected, float tolerance = 0.001f) {
    EXPECT_EQ(actual.getShape(), expected.getShape());
    for (size_t z = 0; z < expected.getShape().depth; z++) {
        for (size_t y = 0; y < expected.getShape().height; y++) {
            for (size_t x = 0; x < expected.getShape().width; x++) {
                EXPECT_NEAR(actual.getPixel(x, y, z), expected.getPixel(x, y, z), tolerance)
                    << "Mismatch at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

void expectSubimageData(const Image3D& subimage, const Image3D& original,
                        size_t ox, size_t oy, size_t oz, float tolerance = 0.001f) {
    CuboidShape subShape = subimage.getShape();
    for (size_t z = 0; z < subShape.depth; z++) {
        for (size_t y = 0; y < subShape.height; y++) {
            for (size_t x = 0; x < subShape.width; x++) {
                float expected = original.getPixel(ox + x, oy + y, oz + z);
                float actual = subimage.getPixel(x, y, z);
                EXPECT_NEAR(actual, expected, tolerance)
                    << "Mismatch at (" << x << "," << y << "," << z << ")"
                    << " offset (" << ox << "," << oy << "," << oz << ")";
            }
        }
    }
}

BoxCoordWithPadding makeBox(size_t ox, size_t oy, size_t oz,
                            size_t w, size_t h, size_t d,
                            size_t pbx = 0, size_t pby = 0, size_t pbz = 0,
                            size_t pax = 0, size_t pay = 0, size_t paz = 0) {
    BoxCoordWithPadding box;
    box.box.position = CuboidPosition(static_cast<int64_t>(ox), static_cast<int64_t>(oy), static_cast<int64_t>(oz));
    box.box.dimensions = CuboidShape(w, h, d);
    box.padding.before = CuboidShape(pbx, pby, pbz);
    box.padding.after = CuboidShape(pax, pay, paz);
    return box;
}

bool hasUninitializedPixels(const Image3D& img) {
    for (auto it = img.cbegin(); it != img.cend(); ++it) {
        if (*it == -1.0f) return true;
    }
    return false;
}

bool isAllZeros(const Image3D& img) {
    for (auto it = img.cbegin(); it != img.cend(); ++it) {
        if (*it != 0.0f) return false;
    }
    return true;
}

} // namespace

// ===== Group 1: Static API — Full-File Data Correctness =====

class TiffReaderStaticTest : public ::testing::Test {
protected:
    std::string testDir;
    void SetUp() override { testDir = TestUtils::outputPath(); }
};

TEST_F(TiffReaderStaticTest, GradientImage_AllPixelsCorrect) {
    Image3D original = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/static_gradient.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, original));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    expectImageEquals(*readOpt, original);
}

TEST_F(TiffReaderStaticTest, RandomImage_AllPixelsCorrect) {
    Image3D original = TestUtils::createRandomImage(32, 24, 12);
    std::string path = testDir + "/static_random.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, original));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    expectImageEquals(*readOpt, original);
}

TEST_F(TiffReaderStaticTest, SingleSlice_AllPixelsCorrect) {
    Image3D original = TestUtils::createGradientImage(8, 8, 1);
    std::string path = testDir + "/static_single_slice.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, original));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    expectImageEquals(*readOpt, original);
}

TEST_F(TiffReaderStaticTest, LargeImage_AllPixelsCorrect) {
    Image3D original = TestUtils::createGradientImage(64, 48, 16);
    std::string path = testDir + "/static_large.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, original));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    expectImageEquals(*readOpt, original);
}

TEST_F(TiffReaderStaticTest, ReadResultNotAllZeros) {
    Image3D original = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/static_notzeros.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, original));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_FALSE(isAllZeros(*readOpt));
    EXPECT_FALSE(hasUninitializedPixels(*readOpt));
}

// ===== Group 2: Instance API — Subimage Data Correctness =====

class TiffReaderSubimageTest : public ::testing::Test {
protected:
    std::string testDir;
    std::string testTiffPath;
    Image3D original;
    size_t w = 32, h = 32, d = 16;

    void SetUp() override {
        testDir = TestUtils::outputPath();
        testTiffPath = testDir + "/subimage_test.tif";
        original = TestUtils::createGradientImage(w, h, d);
        ASSERT_TRUE(TiffWriter::writeToFile(testTiffPath, original));
    }
};

TEST_F(TiffReaderSubimageTest, FullImage_AllPixelsCorrect) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(makeBox(0, 0, 0, w, h, d));
    EXPECT_EQ(result.image.getShape(), CuboidShape(w, h, d));
    expectSubimageData(result.image, original, 0, 0, 0);
}

TEST_F(TiffReaderSubimageTest, AtOffset_AllPixelsCorrect) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(makeBox(8, 8, 4, 16, 16, 8));
    EXPECT_EQ(result.image.getShape(), CuboidShape(16, 16, 8));
    expectSubimageData(result.image, original, 8, 8, 4);
}

TEST_F(TiffReaderSubimageTest, SingleVoxel_CorrectPixel) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(makeBox(15, 10, 7, 1, 1, 1));
    EXPECT_EQ(result.image.getShape(), CuboidShape(1, 1, 1));
    EXPECT_NEAR(result.image.getPixel(0, 0, 0), original.getPixel(15, 10, 7), 0.001f);
}

TEST_F(TiffReaderSubimageTest, AtEdge_AllPixelsCorrect) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(makeBox(28, 28, 14, 4, 4, 2));
    EXPECT_EQ(result.image.getShape(), CuboidShape(4, 4, 2));
    expectSubimageData(result.image, original, 28, 28, 14);
}

TEST_F(TiffReaderSubimageTest, ResultNotAllZeros) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(makeBox(8, 8, 4, 16, 16, 8));
    EXPECT_FALSE(isAllZeros(result.image));
    EXPECT_FALSE(hasUninitializedPixels(result.image));
}

// ===== Group 3: Padding Data Correctness =====

class TiffReaderPaddingTest : public ::testing::Test {
protected:
    std::string testDir;
    std::string testTiffPath;
    Image3D original;
    size_t w = 16, h = 16, d = 8;

    void SetUp() override {
        testDir = TestUtils::outputPath();
        testTiffPath = testDir + "/padding_test.tif";
        original = TestUtils::createGradientImage(w, h, d);
        ASSERT_TRUE(TiffWriter::writeToFile(testTiffPath, original));
    }
};

TEST_F(TiffReaderPaddingTest, WithPadding_CenterDataCorrect) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    size_t pb = 2, pa = 2;
    auto result = reader.getSubimage(
        makeBox(4, 4, 2, 8, 8, 4, pb, pb, pb, pa, pa, pa));

    EXPECT_EQ(result.image.getShape(), CuboidShape(8 + 4, 8 + 4, 4 + 4));

    for (size_t z = 0; z < 4; z++) {
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                float expected = original.getPixel(4 + x, 4 + y, 2 + z);
                float actual = result.image.getPixel(pb + x, pb + y, pb + z);
                EXPECT_NEAR(actual, expected, 0.001f)
                    << "Center mismatch at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

TEST_F(TiffReaderPaddingTest, WithPadding_NoUninitializedPixels) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(
        makeBox(4, 4, 2, 8, 8, 4, 2, 2, 2, 2, 2, 2));

    EXPECT_FALSE(hasUninitializedPixels(result.image));
}

TEST_F(TiffReaderPaddingTest, AtImageEdge_WithPadding_NoCrash) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(
        makeBox(0, 0, 0, 8, 8, 4, 4, 4, 4, 4, 4, 4));

    EXPECT_EQ(result.image.getShape(), CuboidShape(16, 16, 12));

    for (size_t z = 0; z < 4; z++) {
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                float expected = original.getPixel(x, y, z);
                float actual = result.image.getPixel(4 + x, 4 + y, 4 + z);
                EXPECT_NEAR(actual, expected, 0.001f)
                    << "Edge center mismatch at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

TEST_F(TiffReaderPaddingTest, AtFarEdge_WithPadding_NoCrash) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(
        makeBox(8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4));

    EXPECT_EQ(result.image.getShape(), CuboidShape(16, 16, 12));

    for (size_t z = 0; z < 4; z++) {
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                float expected = original.getPixel(8 + x, 8 + y, 4 + z);
                float actual = result.image.getPixel(4 + x, 4 + y, 4 + z);
                EXPECT_NEAR(actual, expected, 0.001f)
                    << "Far edge center mismatch at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

TEST_F(TiffReaderPaddingTest, ZPaddingAtEdge_CenterCorrect) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    auto result = reader.getSubimage(
        makeBox(0, 0, 4, 16, 16, 4, 0, 0, 4, 0, 0, 4));

    EXPECT_EQ(result.image.getShape(), CuboidShape(16, 16, 12));

    for (size_t z = 0; z < 4; z++) {
        for (size_t y = 0; y < 16; y++) {
            for (size_t x = 0; x < 16; x++) {
                float expected = original.getPixel(x, y, 4 + z);
                float actual = result.image.getPixel(x, y, 4 + z);
                EXPECT_NEAR(actual, expected, 0.001f)
                    << "Z-pad center mismatch at (" << x << "," << y << "," << z << ")";
            }
        }
    }
}

// ===== Group 4: Writer setSubimage Round-Trip =====

class TiffWriterSubimageTest : public ::testing::Test {
protected:
    std::string testDir;
    void SetUp() override { testDir = TestUtils::outputPath(); }
};

TEST_F(TiffWriterSubimageTest, FullWrite_ReadBackCorrect) {
    Image3D original = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/writer_full.tif";

    CuboidShape shape = original.getShape();
    {
        TiffWriter writer(path, shape);
        ASSERT_TRUE(writer.setSubimage(original, makeBox(0, 0, 0, shape.width, shape.height, shape.depth).box));
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    expectImageEquals(*readOpt, original);
}

TEST_F(TiffWriterSubimageTest, MultiChunk_WriteReadBackCorrect) {
    size_t w = 16, h = 16, d = 8;
    Image3D original = TestUtils::createGradientImage(w, h, d);
    std::string path = testDir + "/writer_multichunk.tif";

    {
        TiffWriter writer(path, CuboidShape(w, h, d));
        size_t chunkDepth = 2;
        for (size_t z = 0; z < d; z += chunkDepth) {
            size_t thisDepth = std::min(chunkDepth, d - z);

            BoxCoord subBox;
            subBox.position = CuboidPosition(0, 0, static_cast<int64_t>(z));
            subBox.dimensions = CuboidShape(w, h, thisDepth);
            Image3D chunk = original.getSubimageCopy(subBox);

            ASSERT_TRUE(writer.setSubimage(chunk, makeBox(0, 0, z, w, h, thisDepth).box));
        }
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    expectImageEquals(*readOpt, original);
}

TEST_F(TiffWriterSubimageTest, FullWrite_ResultNotAllZeros) {
    Image3D original = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/writer_notzeros.tif";

    CuboidShape shape = original.getShape();
    {
        TiffWriter writer(path, shape);
        ASSERT_TRUE(writer.setSubimage(original, makeBox(0, 0, 0, shape.width, shape.height, shape.depth).box));
    }

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_FALSE(isAllZeros(*readOpt));
    EXPECT_FALSE(hasUninitializedPixels(*readOpt));
}

// ===== Group 5: Concurrency =====

class TiffReaderConcurrencyTest : public ::testing::Test {
protected:
    std::string testDir;
    std::string testTiffPath;
    Image3D original;
    size_t w = 32, h = 32, d = 16;

    void SetUp() override {
        testDir = TestUtils::outputPath();
        testTiffPath = testDir + "/concurrency_test.tif";
        original = TestUtils::createGradientImage(w, h, d);
        ASSERT_TRUE(TiffWriter::writeToFile(testTiffPath, original));
    }
};

TEST_F(TiffReaderConcurrencyTest, ConcurrentGetSubimage_AllCorrect) {
    ReaderConfig config;
    config.numReaderThreads = 4;
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0, config);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);

    struct Req { size_t ox, oy, oz, dw, dh, dd; };
    std::vector<Req> reqs = {
        {0,  0,  0, 16, 16, 8},
        {16, 0,  0, 16, 16, 8},
        {0,  16, 0, 16, 16, 8},
        {16, 16, 0, 16, 16, 8},
        {0,  0,  8, 16, 16, 8},
        {16, 0,  8, 16, 16, 8},
        {0,  16, 8, 16, 16, 8},
        {16, 16, 8, 16, 16, 8},
    };

    std::vector<PaddedImage> results;
    for (const auto& r : reqs) {
        results.push_back(reader.getSubimage(makeBox(r.ox, r.oy, r.oz, r.dw, r.dh, r.dd)));
    }

    for (size_t i = 0; i < reqs.size(); i++) {
        PaddedImage result = results[i];
        const auto& r = reqs[i];
        EXPECT_EQ(result.image.getShape(), CuboidShape(r.dw, r.dh, r.dd));
        expectSubimageData(result.image, original, r.ox, r.oy, r.oz);
    }
}

TEST_F(TiffReaderConcurrencyTest, SequentialGetSubimage_AllCorrect) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);

    struct Req { size_t ox, oy, oz, dw, dh, dd; };
    std::vector<Req> reqs = {
        {0,  0,  0, 16, 16, 8},
        {8,  8,  4, 16, 16, 8},
        {4,  4,  2, 8,  8,  4},
        {16, 16, 8, 16, 16, 8},
    };

    for (const auto& r : reqs) {
        auto result = reader.getSubimage(makeBox(r.ox, r.oy, r.oz, r.dw, r.dh, r.dd));
        EXPECT_EQ(result.image.getShape(), CuboidShape(r.dw, r.dh, r.dd));
        expectSubimageData(result.image, original, r.ox, r.oy, r.oz);
    }
}

// ===== Group 6: Metadata Correctness =====

class TiffReaderMetadataTest : public ::testing::Test {
protected:
    std::string testDir;
    void SetUp() override { testDir = TestUtils::outputPath(); }
};

TEST_F(TiffReaderMetadataTest, AllFieldsCorrect) {
    Image3D img(CuboidShape(12, 10, 6), 1.0f);
    std::string path = testDir + "/metadata_test.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, img));

    auto metaOpt = TiffReader::readMetadata(path);
    ASSERT_TRUE(metaOpt.has_value());
    EXPECT_EQ(metaOpt->imageWidth, 12);
    EXPECT_EQ(metaOpt->imageLength, 10);
    EXPECT_EQ(metaOpt->slices, 6);
    EXPECT_EQ(metaOpt->bitsPerSample, 32);
    EXPECT_EQ(metaOpt->samplesPerPixel, 1);
    EXPECT_EQ(metaOpt->sampleFormat, SAMPLEFORMAT_IEEEFP);
    EXPECT_FALSE(metaOpt->isTiled);
    EXPECT_GT(metaOpt->rowsPerStrip, 0);
}

TEST_F(TiffReaderMetadataTest, InstanceReader_MetadataMatchesStatic) {
    Image3D img = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/metadata_instance.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, img));

    auto staticMeta = TiffReader::readMetadata(path);
    ASSERT_TRUE(staticMeta.has_value());

    auto tr = std::make_unique<TiffReader>(path);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    const auto& instanceMeta = reader.getMetaData();

    EXPECT_EQ(instanceMeta.imageWidth, staticMeta->imageWidth);
    EXPECT_EQ(instanceMeta.imageLength, staticMeta->imageLength);
    EXPECT_EQ(instanceMeta.slices, staticMeta->slices);
    EXPECT_EQ(instanceMeta.bitsPerSample, staticMeta->bitsPerSample);
    EXPECT_EQ(instanceMeta.sampleFormat, staticMeta->sampleFormat);
}
