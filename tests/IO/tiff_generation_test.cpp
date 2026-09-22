#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/IO/TiffReader.h"
#include "TestUtils.h"

class LoggingEnvironment : public ::testing::Environment {
public:
    void SetUp() override { TestUtils::initLogging(); }
};
inline ::testing::Environment* logEnv = ::testing::AddGlobalTestEnvironment(new LoggingEnvironment());

class TiffGenerationTest : public ::testing::Test {
protected:
    std::string testDir;

    void SetUp() override {
        testDir = TestUtils::outputPath();
    }
};

TEST_F(TiffGenerationTest, GenerateConstantZero) {
    Image3D img(CuboidShape(32, 32, 20), 0.0f);
    std::string path = testDir + "/constant_zero.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, img));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), CuboidShape(32, 32, 20));
    for (auto it = readOpt->cbegin(); it != readOpt->cend(); ++it) {
        EXPECT_FLOAT_EQ(*it, 0.0f);
    }
}

TEST_F(TiffGenerationTest, GenerateConstantOne) {
    Image3D img(CuboidShape(32, 32, 20), 1.0f);
    std::string path = testDir + "/constant_one.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, img));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), CuboidShape(32, 32, 20));
    for (auto it = readOpt->cbegin(); it != readOpt->cend(); ++it) {
        EXPECT_NEAR(*it, 1.0f, 0.001f);
    }
}

TEST_F(TiffGenerationTest, GenerateGradientImage) {
    Image3D img = TestUtils::createGradientImage(16, 16, 8);
    std::string path = testDir + "/gradient.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, img));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_TRUE(readOpt->isEqual(img, 0.001f));
}

TEST_F(TiffGenerationTest, GenerateSmallImage) {
    Image3D img(CuboidShape(4, 4, 2), 7.0f);
    std::string path = testDir + "/small.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(path, img));

    auto readOpt = TiffReader::readTiffFile(path, 0);
    ASSERT_TRUE(readOpt.has_value());
    EXPECT_EQ(readOpt->getShape(), CuboidShape(4, 4, 2));
}
