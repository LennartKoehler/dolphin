#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "TestUtils.h"

class ImageComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        img1 = Image3D(CuboidShape(4, 4, 2), 0.0f);
        for (size_t i = 0; i < 4 * 4 * 2; i++) {
            img1[i] = static_cast<float>(i + 1);
        }
        img2 = img1;
    }

    Image3D img1;
    Image3D img2;
};

TEST_F(ImageComparisonTest, IdenticalImagesAreEqual) {
    EXPECT_TRUE(img1.isEqual(img2, 0.0f));
}

TEST_F(ImageComparisonTest, DifferentImagesAreNotEqual) {
    img2.setPixel(0, 0, 0, 999.0f);
    EXPECT_FALSE(img1.isEqual(img2, 0.0f));
}

TEST_F(ImageComparisonTest, EqualWithinTolerance) {
    img2.setPixel(0, 0, 0, img1.getPixel(0, 0, 0) + 0.005f);
    EXPECT_TRUE(img1.isEqual(img2, 0.01f));
    EXPECT_FALSE(img1.isEqual(img2, 0.001f));
}

TEST_F(ImageComparisonTest, DifferentShapeNotEqual) {
    Image3D different(CuboidShape(2, 2, 2), 1.0f);
    EXPECT_FALSE(img1.isEqual(different, 0.0f));
}

TEST_F(ImageComparisonTest, SinglePixelDifference) {
    img2.setPixel(3, 3, 1, 0.0f);
    EXPECT_FALSE(img1.isEqual(img2, 0.0f));
}

TEST_F(ImageComparisonTest, ZeroToleranceExactMatch) {
    EXPECT_TRUE(img1.isEqual(img2, 0.0f));
}

TEST_F(ImageComparisonTest, LargeTolerance) {
    img2.setPixel(0, 0, 0, 500.0f);
    EXPECT_TRUE(img1.isEqual(img2, 1000.0f));
}

TEST_F(ImageComparisonTest, CompareWithSelf) {
    EXPECT_TRUE(img1.isEqual(img1, 0.0f));
}
