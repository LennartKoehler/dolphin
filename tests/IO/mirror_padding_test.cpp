#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/ReaderWriter.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "TestUtils.h"

class MirrorPaddingTest : public ::testing::Test {
protected:
    std::string testDir;
    std::string testTiffPath;

    void SetUp() override {
        testDir = TestUtils::outputPath();
        testTiffPath = testDir + "/mirror_test_input.tif";

        Image3D img(CuboidShape(16, 16, 8), 0.0f);
        for (size_t i = 0; i < 16 * 16 * 8; i++) {
            img[i] = static_cast<float>(i);
        }
        TiffWriter::writeToFile(testTiffPath, img);
    }
};

TEST_F(MirrorPaddingTest, ReadSubimageNoPadding) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    BoxCoordWithPadding box;
    box.box.position = CuboidShape(0, 0, 0);
    box.box.dimensions = CuboidShape(8, 8, 4);
    box.padding.before = CuboidShape(0, 0, 0);
    box.padding.after = CuboidShape(0, 0, 0);

    auto result = reader.getSubimage(box);
    EXPECT_EQ(result.image.getShape(), CuboidShape(8, 8, 4));
}

TEST_F(MirrorPaddingTest, ReadSubimageWithPadding) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    BoxCoordWithPadding box;
    box.box.position = CuboidShape(0, 0, 0);
    box.box.dimensions = CuboidShape(8, 8, 4);
    box.padding.before = CuboidShape(2, 2, 2);
    box.padding.after = CuboidShape(2, 2, 2);

    auto result = reader.getSubimage(box);
    EXPECT_EQ(result.image.getShape(), CuboidShape(12, 12, 8));
}

TEST_F(MirrorPaddingTest, ReadFullImage) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    BoxCoordWithPadding box;
    box.box.position = CuboidShape(0, 0, 0);
    box.box.dimensions = CuboidShape(16, 16, 8);
    box.padding.before = CuboidShape(0, 0, 0);
    box.padding.after = CuboidShape(0, 0, 0);

    auto result = reader.getSubimage(box);
    EXPECT_EQ(result.image.getShape(), CuboidShape(16, 16, 8));
}

TEST_F(MirrorPaddingTest, ReadSubimageAtOffset) {
    auto tr = std::make_unique<TiffReader>(testTiffPath);
    tr->configure(0);
    ReaderHandler reader(std::move(tr), PaddingFillType::MIRROR);
    BoxCoordWithPadding box;
    box.box.position = CuboidShape(4, 4, 2);
    box.box.dimensions = CuboidShape(8, 8, 4);
    box.padding.before = CuboidShape(0, 0, 0);
    box.padding.after = CuboidShape(0, 0, 0);

    auto result = reader.getSubimage(box);
    EXPECT_EQ(result.image.getShape(), CuboidShape(8, 8, 4));
}

TEST_F(MirrorPaddingTest, ReadMetadata) {
    auto meta = TiffReader::readMetadata(testTiffPath);
    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->imageWidth, 16);
    EXPECT_EQ(meta->imageLength, 16);
    EXPECT_EQ(meta->slices, 8);
}
