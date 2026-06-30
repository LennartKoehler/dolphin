#include <gtest/gtest.h>
#include "dolphin_image/Image3D.h"
#include "TestUtils.h"
#include <cmath>

class Image3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        img3x3x3 = TestUtils::createConstantImage(3, 3, 3, 0.0f);
        for (int i = 0; i < 27; i++) {
            img3x3x3[i] = static_cast<float>(i);
        }
    }

    Image3D img3x3x3;
};

TEST_F(Image3DTest, ConstructFromShapeAndValue) {
    Image3D img(CuboidShape(5, 6, 7), 1.5f);
    EXPECT_EQ(img.getShape(), CuboidShape(5, 6, 7));
    for (auto it = img.cbegin(); it != img.cend(); ++it) {
        EXPECT_FLOAT_EQ(*it, 1.5f);
    }
}

TEST_F(Image3DTest, CopyConstructor) {
    Image3D copy(img3x3x3);
    EXPECT_TRUE(copy.isEqual(img3x3x3, 0.0f));
}

TEST_F(Image3DTest, MoveConstructor) {
    Image3D moved(std::move(img3x3x3));
    EXPECT_EQ(moved.getShape(), CuboidShape(3, 3, 3));
}

TEST_F(Image3DTest, AssignmentOperator) {
    Image3D assigned;
    assigned = img3x3x3;
    EXPECT_TRUE(assigned.isEqual(img3x3x3, 0.0f));
}

TEST_F(Image3DTest, GetShape) {
    EXPECT_EQ(img3x3x3.getShape(), CuboidShape(3, 3, 3));
}

TEST_F(Image3DTest, GetSetPixel) {
    img3x3x3.setPixel(1, 2, 0, 42.0f);
    EXPECT_FLOAT_EQ(img3x3x3.getPixel(1, 2, 0), 42.0f);
}

TEST_F(Image3DTest, GetSetPixelOrigin) {
    img3x3x3.setPixel(0, 0, 0, 99.0f);
    EXPECT_FLOAT_EQ(img3x3x3.getPixel(0, 0, 0), 99.0f);
}

TEST_F(Image3DTest, GetSetPixelMaxCorner) {
    img3x3x3.setPixel(2, 2, 2, 77.0f);
    EXPECT_FLOAT_EQ(img3x3x3.getPixel(2, 2, 2), 77.0f);
}

TEST_F(Image3DTest, OperatorIndex) {
    img3x3x3[13] = 55.0f;
    EXPECT_FLOAT_EQ(img3x3x3[13], 55.0f);
}

TEST_F(Image3DTest, GetMax) {
    Image3D img(CuboidShape(3, 3, 3), 0.0f);
    img.setPixel(0, 0, 0, 1.0f);
    img.setPixel(1, 1, 1, 5.0f);
    img.setPixel(2, 2, 2, 3.0f);
    EXPECT_FLOAT_EQ(img.getMax(), 5.0f);
}

TEST_F(Image3DTest, IsEqualExact) {
    Image3D copy(img3x3x3);
    EXPECT_TRUE(img3x3x3.isEqual(copy, 0.0f));
}

TEST_F(Image3DTest, IsEqualDifferent) {
    Image3D copy(img3x3x3);
    copy.setPixel(0, 0, 0, 999.0f);
    EXPECT_FALSE(img3x3x3.isEqual(copy, 0.0f));
}

TEST_F(Image3DTest, IsEqualWithinTolerance) {
    Image3D copy(img3x3x3);
    copy.setPixel(0, 0, 0, img3x3x3.getPixel(0, 0, 0) + 0.001f);
    EXPECT_TRUE(img3x3x3.isEqual(copy, 0.01f));
    EXPECT_FALSE(img3x3x3.isEqual(copy, 0.0001f));
}

TEST_F(Image3DTest, GetInRange) {
    Image3D img(CuboidShape(3, 1, 1), 0.0f);
    img.setPixel(0, 0, 0, 0.5f);
    img.setPixel(1, 0, 0, 1.5f);
    img.setPixel(2, 0, 0, 2.5f);
    Image3D result = img.getInRange(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(result.getPixel(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(result.getPixel(1, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result.getPixel(2, 0, 0), 0.0f);
}

TEST_F(Image3DTest, GetSubimageCopy) {
    BoxCoord coord;
    coord.position = CuboidShape(0, 0, 0);
    coord.dimensions = CuboidShape(2, 2, 2);
    Image3D sub = img3x3x3.getSubimageCopy(coord);
    EXPECT_EQ(sub.getShape(), CuboidShape(2, 2, 2));
    EXPECT_FLOAT_EQ(sub.getPixel(0, 0, 0), img3x3x3.getPixel(0, 0, 0));
}

TEST_F(Image3DTest, GetSubimageCopyOffset) {
    BoxCoord coord;
    coord.position = CuboidShape(1, 1, 1);
    coord.dimensions = CuboidShape(2, 2, 2);
    Image3D sub = img3x3x3.getSubimageCopy(coord);
    EXPECT_EQ(sub.getShape(), CuboidShape(2, 2, 2));
    EXPECT_FLOAT_EQ(sub.getPixel(0, 0, 0), img3x3x3.getPixel(1, 1, 1));
}

TEST_F(Image3DTest, IteratorBasic) {
    Image3D img(CuboidShape(2, 2, 2), 0.0f);
    int count = 0;
    for (auto it = img.begin(); it != img.end(); ++it) {
        *it = static_cast<float>(count);
        count++;
    }
    EXPECT_EQ(count, 8);
    EXPECT_FLOAT_EQ(img[0], 0.0f);
    EXPECT_FLOAT_EQ(img[7], 7.0f);
}

TEST_F(Image3DTest, IteratorGetCoordinates) {
    Image3D img(CuboidShape(3, 3, 3), 0.0f);
    auto it = img.begin();
    int x, y, z;
    it.getCoordinates(x, y, z);
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 0);
    EXPECT_EQ(z, 0);
}

TEST_F(Image3DTest, ConstIterator) {
    Image3D img(CuboidShape(2, 2, 2), 3.14f);
    int count = 0;
    for (auto it = img.cbegin(); it != img.cend(); ++it) {
        EXPECT_FLOAT_EQ(*it, 3.14f);
        count++;
    }
    EXPECT_EQ(count, 8);
}

TEST_F(Image3DTest, Flip) {
    Image3D img(CuboidShape(2, 1, 1), 0.0f);
    img.setPixel(0, 0, 0, 1.0f);
    img.setPixel(1, 0, 0, 2.0f);
    img.flip();
    float v0 = img.getPixel(0, 0, 0);
    float v1 = img.getPixel(1, 0, 0);
    EXPECT_NE(v0, v1);
}

TEST_F(Image3DTest, GetItkImage) {
    Image3D img(CuboidShape(2, 2, 2), 1.0f);
    auto itkImg = img.getItkImage();
    ASSERT_NE(itkImg, nullptr);
    auto size = itkImg->GetLargestPossibleRegion().GetSize();
    EXPECT_EQ(size[0], 2);
    EXPECT_EQ(size[1], 2);
    EXPECT_EQ(size[2], 2);
}

TEST_F(Image3DTest, SetItkImage) {
    auto itkImg = ImageType::New();
    ImageType::SizeType size = {4, 4, 4};
    ImageType::RegionType region;
    region.SetSize(size);
    itkImg->SetRegions(region);
    itkImg->Allocate();
    itkImg->FillBuffer(7.0f);

    Image3D img;
    img.setItkImage(itkImg);
    EXPECT_EQ(img.getShape(), CuboidShape(4, 4, 4));
    EXPECT_FLOAT_EQ(img.getPixel(0, 0, 0), 7.0f);
}

TEST_F(Image3DTest, EmptyImage) {
    Image3D img;
    auto shape = img.getShape();
    EXPECT_EQ(shape.width, 0);
    EXPECT_EQ(shape.height, 0);
    EXPECT_EQ(shape.depth, 0);
}

TEST_F(Image3DTest, HasNaN) {
    Image3D img(CuboidShape(2, 2, 2), 1.0f);
    EXPECT_FALSE(TestUtils::hasNaN(img));
    img.setPixel(0, 0, 0, std::nanf(""));
    EXPECT_TRUE(TestUtils::hasNaN(img));
}

TEST_F(Image3DTest, HasInf) {
    Image3D img(CuboidShape(2, 2, 2), 1.0f);
    EXPECT_FALSE(TestUtils::hasInf(img));
    img.setPixel(0, 0, 0, std::numeric_limits<float>::infinity());
    EXPECT_TRUE(TestUtils::hasInf(img));
}

TEST_F(Image3DTest, GetRegionLargerThreshold) {
    Image3D img(CuboidShape(5, 5, 5), 0.0f);
    img.setPixel(2, 2, 2, 1.0f);
    img.setPixel(0, 0, 0, 0.001f);
    auto region = img.getRegionLargerThreshold(0.01f);
    EXPECT_GE(region.width, 1);
    EXPECT_GE(region.height, 1);
    EXPECT_GE(region.depth, 1);
}
