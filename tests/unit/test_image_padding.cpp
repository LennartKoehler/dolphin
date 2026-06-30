#include <gtest/gtest.h>
#include "dolphin_image/ImagePadding.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "TestUtils.h"

class ImagePaddingTest : public ::testing::Test {
protected:
    Image3D img{CuboidShape(8, 8, 8), 1.0f};

    void SetUp() override {
        for (int z = 0; z < 8; z++) {
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    img.setPixel(x, y, z, static_cast<float>(x + y * 8 + z * 64));
                }
            }
        }
    }
};

TEST_F(ImagePaddingTest, PadImageZero) {
    Padding padding;
    padding.before = CuboidShape(2, 2, 2);
    padding.after = CuboidShape(2, 2, 2);
    Image3D padded = img;
    ImagePadding::padImageZero(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
    EXPECT_FLOAT_EQ(padded.getPixel(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(padded.getPixel(2, 2, 2), img.getPixel(0, 0, 0));
}

TEST_F(ImagePaddingTest, PadImageMirror) {
    Padding padding;
    padding.before = CuboidShape(2, 0, 0);
    padding.after = CuboidShape(2, 0, 0);
    Image3D padded = img;
    ImagePadding::padImageMirror(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 8, 8));
    EXPECT_FLOAT_EQ(padded.getPixel(0, 0, 0), img.getPixel(1, 0, 0));
    EXPECT_FLOAT_EQ(padded.getPixel(1, 0, 0), img.getPixel(0, 0, 0));
    EXPECT_FLOAT_EQ(padded.getPixel(2, 0, 0), img.getPixel(0, 0, 0));
}

TEST_F(ImagePaddingTest, PadImageLinear) {
    Padding padding;
    padding.before = CuboidShape(2, 0, 0);
    padding.after = CuboidShape(2, 0, 0);
    Image3D padded = img;
    ImagePadding::padImageLinear(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 8, 8));
}

TEST_F(ImagePaddingTest, PadImageQuadratic) {
    Padding padding;
    padding.before = CuboidShape(2, 2, 2);
    padding.after = CuboidShape(2, 2, 2);
    Image3D padded = img;
    ImagePadding::padImageQuadratic(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(ImagePaddingTest, PadImageSinusoid) {
    Padding padding;
    padding.before = CuboidShape(2, 2, 2);
    padding.after = CuboidShape(2, 2, 2);
    Image3D padded = img;
    ImagePadding::padImageSinusoid(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(ImagePaddingTest, PadImageGaussian) {
    Padding padding;
    padding.before = CuboidShape(2, 2, 2);
    padding.after = CuboidShape(2, 2, 2);
    Image3D padded = img;
    ImagePadding::padImageGaussian(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(ImagePaddingTest, PadImageGeneric) {
    Padding padding;
    padding.before = CuboidShape(1, 1, 1);
    padding.after = CuboidShape(1, 1, 1);
    Image3D padded = img;
    ImagePadding::padImage(padded, padding, PaddingFillType::ZERO);
    EXPECT_EQ(padded.getShape(), CuboidShape(10, 10, 10));
}

TEST_F(ImagePaddingTest, PadToShape) {
    Image3D padded = img;
    auto padding = ImagePadding::padToShape(padded, CuboidShape(12, 12, 12), PaddingFillType::ZERO);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(ImagePaddingTest, ExpandToMinSize) {
    Image3D small(CuboidShape(4, 4, 4), 5.0f);
    ImagePadding::expandToMinSize(small, CuboidShape(8, 8, 8));
    EXPECT_EQ(small.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(ImagePaddingTest, ExpandToMinSizeAlreadyLarge) {
    Image3D large(CuboidShape(16, 16, 16), 1.0f);
    ImagePadding::expandToMinSize(large, CuboidShape(8, 8, 8));
    EXPECT_EQ(large.getShape(), CuboidShape(16, 16, 16));
}

TEST_F(ImagePaddingTest, ZeroPadding) {
    Padding padding;
    padding.before = CuboidShape(0, 0, 0);
    padding.after = CuboidShape(0, 0, 0);
    Image3D padded = img;
    ImagePadding::padImageZero(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(ImagePaddingTest, AsymmetricPadding) {
    Padding padding;
    padding.before = CuboidShape(3, 1, 0);
    padding.after = CuboidShape(1, 3, 2);
    Image3D padded = img;
    ImagePadding::padImageZero(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 10));
}
