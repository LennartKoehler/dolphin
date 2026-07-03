#include <gtest/gtest.h>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin_image/Types/PaddingFillType.h"
#include "TestUtils.h"

class PaddingTest : public ::testing::Test {
protected:
    Image3D img{CuboidShape(8, 8, 8), 0.0f};

    void SetUp() override {
        for (size_t i = 0; i < 8 * 8 * 8; i++) {
            img[i] = static_cast<float>(i);
        }
    }

    Padding makePadding(size_t w, size_t h, size_t d) {
        Padding p;
        p.before = CuboidShape(w, h, d);
        p.after = CuboidShape(w, h, d);
        return p;
    }
};

TEST_F(PaddingTest, MirrorPadding) {
    auto padding = makePadding(2, 2, 2);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::MIRROR);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
    EXPECT_FLOAT_EQ(padded.getPixel(2, 2, 2), img.getPixel(0, 0, 0));
}

TEST_F(PaddingTest, ZeroPadding) {
    auto padding = makePadding(2, 2, 2);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::ZERO);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
    EXPECT_FLOAT_EQ(padded.getPixel(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(padded.getPixel(2, 2, 2), img.getPixel(0, 0, 0));
}

TEST_F(PaddingTest, LinearPadding) {
    auto padding = makePadding(2, 0, 0);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::LINEAR);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 8, 8));
}

TEST_F(PaddingTest, QuadraticPadding) {
    auto padding = makePadding(2, 2, 2);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::QUADRATIC);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(PaddingTest, SinusoidPadding) {
    auto padding = makePadding(2, 2, 2);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::SINUSOID);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(PaddingTest, GaussianPadding) {
    auto padding = makePadding(2, 2, 2);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::GAUSSIAN);
    EXPECT_EQ(padded.getShape(), CuboidShape(12, 12, 12));
}

TEST_F(PaddingTest, NoPadding) {
    auto padding = makePadding(0, 0, 0);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::ZERO);
    EXPECT_EQ(padded.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(PaddingTest, AsymmetricPadding) {
    Padding padding;
    padding.before = CuboidShape(1, 2, 3);
    padding.after = CuboidShape(4, 5, 6);
    Image3D padded = img;
    Preprocessor::padImage(padded, padding, PaddingFillType::ZERO);
    EXPECT_EQ(padded.getShape(), CuboidShape(13, 15, 17));
}

TEST_F(PaddingTest, PadToShape) {
    Image3D padded = img;
    auto padding = Preprocessor::padToShape(padded, CuboidShape(16, 16, 16), PaddingFillType::ZERO);
    EXPECT_EQ(padded.getShape(), CuboidShape(16, 16, 16));
}

TEST_F(PaddingTest, ExpandToMinSize) {
    Image3D small(CuboidShape(4, 4, 4), 1.0f);
    Preprocessor::expandToMinSize(small, CuboidShape(16, 16, 16));
    EXPECT_EQ(small.getShape(), CuboidShape(16, 16, 16));
}
