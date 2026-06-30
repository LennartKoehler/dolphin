#include <gtest/gtest.h>
#include "dolphin_image/ImageOperations.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "TestUtils.h"

class ImageOperationsTest : public ::testing::Test {
protected:
    Image3D img{CuboidShape(8, 8, 8), 0.0f};

    void SetUp() override {
        for (int i = 0; i < 8 * 8 * 8; i++) {
            img[i] = static_cast<float>(i);
        }
    }
};

TEST_F(ImageOperationsTest, InsertCubeInImage) {
    Image3D cube(CuboidShape(4, 4, 4), 99.0f);
    BoxCoord cubeBox;
    cubeBox.position = CuboidShape(0, 0, 0);
    cubeBox.dimensions = CuboidShape(4, 4, 4);
    BoxCoord srcBox;
    srcBox.position = CuboidShape(2, 2, 2);
    srcBox.dimensions = CuboidShape(4, 4, 4);

    ImageOperations::insertCubeInImage(cube, cubeBox, img, srcBox);

    EXPECT_FLOAT_EQ(img.getPixel(2, 2, 2), 99.0f);
    EXPECT_FLOAT_EQ(img.getPixel(5, 5, 5), 99.0f);
    EXPECT_NE(img.getPixel(0, 0, 0), 99.0f);
}

TEST_F(ImageOperationsTest, AddCubeToImage) {
    Image3D cube(CuboidShape(8, 8, 8), 1.0f);
    float original = img.getPixel(0, 0, 0);
    ImageOperations::addCubeToImage(cube, img);
    EXPECT_FLOAT_EQ(img.getPixel(0, 0, 0), original + 1.0f);
}

TEST_F(ImageOperationsTest, RemovePadding) {
    Image3D padded(CuboidShape(12, 12, 12), 0.0f);
    for (int z = 2; z < 10; z++) {
        for (int y = 2; y < 10; y++) {
            for (int x = 2; x < 10; x++) {
                padded.setPixel(x, y, z, 1.0f);
            }
        }
    }

    Padding padding;
    padding.before = CuboidShape(2, 2, 2);
    padding.after = CuboidShape(2, 2, 2);
    ImageOperations::removePadding(padded, padding);
    EXPECT_EQ(padded.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(ImageOperationsTest, CropToOriginalSize) {
    Image3D padded(CuboidShape(16, 16, 16), 5.0f);
    ImageOperations::cropToOriginalSize(padded, CuboidShape(8, 8, 8));
    EXPECT_EQ(padded.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(ImageOperationsTest, NormalizeChannel) {
    Image3D image(CuboidShape(4, 4, 4), 0.0f);
    image.setPixel(0, 0, 0, 0.0f);
    image.setPixel(1, 0, 0, 10.0f);
    image.setPixel(2, 0, 0, 20.0f);

    ImageOperations::normalizeChannel(image);

    float maxVal = image.getMax();
    EXPECT_LE(maxVal, 1.0f + 1e-5f);
}

TEST_F(ImageOperationsTest, InsertCubeAtOrigin) {
    Image3D cube(CuboidShape(4, 4, 4), 7.0f);
    Image3D target(CuboidShape(8, 8, 8), 0.0f);
    BoxCoord cubeBox;
    cubeBox.position = CuboidShape(0, 0, 0);
    cubeBox.dimensions = CuboidShape(4, 4, 4);
    BoxCoord srcBox = cubeBox;

    ImageOperations::insertCubeInImage(cube, cubeBox, target, srcBox);
    EXPECT_FLOAT_EQ(target.getPixel(0, 0, 0), 7.0f);
    EXPECT_FLOAT_EQ(target.getPixel(3, 3, 3), 7.0f);
    EXPECT_FLOAT_EQ(target.getPixel(4, 4, 4), 0.0f);
}

TEST_F(ImageOperationsTest, AddCubeToImagePartial) {
    Image3D target(CuboidShape(4, 4, 4), 10.0f);
    Image3D cube(CuboidShape(4, 4, 4), 5.0f);
    ImageOperations::addCubeToImage(cube, target);
    EXPECT_FLOAT_EQ(target.getPixel(0, 0, 0), 15.0f);
    EXPECT_FLOAT_EQ(target.getPixel(3, 3, 3), 15.0f);
}
