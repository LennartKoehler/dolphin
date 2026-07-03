#include <gtest/gtest.h>
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin/Logging.h"
#include <algorithm>

class ImageSplitTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

std::string paddingStrategyToString(PaddingStrategyType strategy) {
    switch (strategy) {
        case PaddingStrategyType::NONE:     return "NONE";
        case PaddingStrategyType::PARENT:   return "PARENT";
        case PaddingStrategyType::FULL_PSF: return "FULL_PSF";
        case PaddingStrategyType::MANUAL:   return "MANUAL";
        default:                            return "UNKNOWN";
    }
}

std::string printShape(const CuboidShape& imageSize,
                       const Padding& padding,
                       const size_t maxVolumePerCube,
                       const size_t minNumberCubes,
                       const PaddingStrategyType& paddingStrategy,
                       const CuboidShape& minShape,
                       const Result<std::vector<BoxCoordWithPadding>>& result) {
    std::string s = "imageSize: " + imageSize.print()
        + ", padding: before " + padding.before.print() + " / after " + padding.after.print()
        + ", maxVolumePerCube: " + std::to_string(maxVolumePerCube)
        + ", minNumberCubes: " + std::to_string(minNumberCubes)
        + ", paddingStrategy: " + paddingStrategyToString(paddingStrategy)
        + ", minShape: " + minShape.print();
    if (!result.success) {
        s += " -> FAILED: " + result.getErrorString();
    } else {
        s += " -> " + std::to_string(result.value.size()) + " cubes of size " + result.value[0].box.dimensions.print();
    }
    return s;
}

TEST_F(ImageSplitTest, SmallImageNoPadding) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imageSize, 1000000, 1, PaddingStrategyType::NONE, CuboidShape(1, 1, 1));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 1000000, 1, PaddingStrategyType::NONE, CuboidShape(1, 1, 1), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, ImageWithPadding) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, imageSize, 1000000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 1000000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, LargeImageMultipleCubes) {
    CuboidShape imageSize(512, 512, 100);
    Padding padding{CuboidShape(32, 32, 16), CuboidShape(32, 32, 16)};
    auto result = splitImageHomogeneous(padding, imageSize, 1000000, 4, PaddingStrategyType::FULL_PSF, CuboidShape(65, 65, 33));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 1000000, 4, PaddingStrategyType::FULL_PSF, CuboidShape(65, 65, 33), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 4u);
}


TEST_F(ImageSplitTest, VeryLargeImage) {
    CuboidShape imageSize(2000, 13000, 200);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    auto result = splitImageHomogeneous(padding, imageSize, 9e9, 8, PaddingStrategyType::NONE, CuboidShape(9, 9, 5));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 9e9, 8, PaddingStrategyType::NONE, CuboidShape(9, 9, 5), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, VerySmallImage) {
    CuboidShape imageSize(32, 32, 10);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    auto result = splitImageHomogeneous(padding, imageSize, 1000000, 1, PaddingStrategyType::NONE, CuboidShape(9, 9, 5));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 1000000, 1, PaddingStrategyType::NONE, CuboidShape(9, 9, 5), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, NonePaddingType) {
    CuboidShape imageSize(200, 200, 80);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    auto result = splitImageHomogeneous(padding, imageSize, 500000, 2, PaddingStrategyType::NONE, CuboidShape(41, 41, 21));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 500000, 2, PaddingStrategyType::NONE, CuboidShape(41, 41, 21), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, CubeCoverageVerification) {
    CuboidShape imageSize(128, 128, 32);
    Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
    auto result = splitImageHomogeneous(padding, imageSize, 500000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(17, 17, 9));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 500000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(17, 17, 9), result);
    ASSERT_TRUE(result.success);
    for (const auto& cube : result.value) {
        EXPECT_GE(cube.box.position.width, 0);
        EXPECT_GE(cube.box.position.height, 0);
        EXPECT_GE(cube.box.position.depth, 0);
        EXPECT_GT(cube.box.dimensions.width, 0);
        EXPECT_GT(cube.box.dimensions.height, 0);
        EXPECT_GT(cube.box.dimensions.depth, 0);
    }
}

TEST_F(ImageSplitTest, HighMinCubes) {
    CuboidShape imageSize(256, 256, 64);
    Padding padding{CuboidShape(16, 16, 8), CuboidShape(16, 16, 8)};
    auto result = splitImageHomogeneous(padding, imageSize, 1000000, 8, PaddingStrategyType::FULL_PSF, CuboidShape(33, 33, 17));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 1000000, 8, PaddingStrategyType::FULL_PSF, CuboidShape(33, 33, 17), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 8u);
}

TEST_F(ImageSplitTest, ConstrainedMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, imageSize, 50000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 50000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, AsymmetricImage) {
    CuboidShape imageSize(400, 200, 30);
    Padding padding{CuboidShape(20, 10, 5), CuboidShape(20, 10, 5)};
    auto result = splitImageHomogeneous(padding, imageSize, 300000, 2, PaddingStrategyType::NONE, CuboidShape(41, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 300000, 2, PaddingStrategyType::NONE, CuboidShape(41, 21, 11), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, FailTooLittleMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, imageSize, 500, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 500, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11), result);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageSplitTest, FailTooManyCubes) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, imageSize, 5000, 100, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 5000, 100, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11), result);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageSplitTest, FirstCubeAtOriginNonePadding) {
    CuboidShape imageSize(128, 128, 32);
    Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
    auto result = splitImageHomogeneous(padding, imageSize, 500000, 1, PaddingStrategyType::NONE, CuboidShape(17, 17, 9));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 500000, 1, PaddingStrategyType::NONE, CuboidShape(17, 17, 9), result);
    ASSERT_TRUE(result.success);

    bool foundOriginCube = false;
    for (const auto& cube : result.value) {
        if (cube.box.position.width == 0 && cube.box.position.height == 0 && cube.box.position.depth == 0) {
            foundOriginCube = true;
            EXPECT_EQ(cube.padding.before.width, 0);
            EXPECT_EQ(cube.padding.before.height, 0);
            EXPECT_EQ(cube.padding.before.depth, 0);
            break;
        }
    }
    EXPECT_TRUE(foundOriginCube);
}

TEST_F(ImageSplitTest, MinimumViableImage) {
    CuboidShape imageSize(10, 10, 10);
    Padding padding{CuboidShape(2, 2, 2), CuboidShape(2, 2, 2)};
    auto result = splitImageHomogeneous(padding, imageSize, 1000000, 1, PaddingStrategyType::NONE, CuboidShape(5, 5, 5));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 1000000, 1, PaddingStrategyType::NONE, CuboidShape(5, 5, 5), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, NoPadding) {
    CuboidShape imageSize(64, 64, 32);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imageSize, 500000, 1, PaddingStrategyType::NONE, CuboidShape(1, 1, 1));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 500000, 1, PaddingStrategyType::NONE, CuboidShape(1, 1, 1), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, LargePaddingRelative) {
    CuboidShape imageSize(50, 50, 20);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    auto result = splitImageHomogeneous(padding, imageSize, 500000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(41, 41, 21));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 500000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(41, 41, 21), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

TEST_F(ImageSplitTest, CubeSizeEqualsImage) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, imageSize, 10000000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, 10000000, 1, PaddingStrategyType::FULL_PSF, CuboidShape(21, 21, 11), result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 1u);
}

