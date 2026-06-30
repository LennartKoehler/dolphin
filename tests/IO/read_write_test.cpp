#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "TestUtils.h"
#include <filesystem>

class ReadWriteTest : public ::testing::Test {
protected:
    std::string testDir;

    void SetUp() override {
        testDir = TestUtils::outputPath();
    }
};

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
