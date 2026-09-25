/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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
bool checkSameShape(const CuboidShape& fullImage, const std::vector<BoxCoordWithPadding>& cubes){
    if (cubes.empty()) return false;
    CuboidShape shape = cubes[0].getPaddedBox().dimensions;
    for (const BoxCoordWithPadding& cube : cubes){
       if (cube.getPaddedBox().dimensions != shape) return false;
    }
    return true;
}

constexpr size_t maxCompletenessCheckVolume = 100000000; // ~400 MB as float image

bool checkCompleteness(const CuboidShape& fullImage, const std::vector<BoxCoordWithPadding>& cubes){
    if (cubes.empty()) return false;
    if (fullImage.getVolume() > maxCompletenessCheckVolume) {
        GTEST_LOG_(WARNING) << "Skipping completeness check, image too large: " << fullImage.print();
        return true;
    }

    Image3D image(fullImage, 1.0);
    const BoxCoord imageBox{CuboidShape(0, 0, 0), fullImage};

    for (const BoxCoordWithPadding& cube : cubes){
        // boxes may extend past the image when boundary padding is absorbed into the box
        BoxCoord visibleBox = cube.box;
        visibleBox.cropTo(imageBox);
        if (visibleBox.dimensions.width == 0 || visibleBox.dimensions.height == 0 || visibleBox.dimensions.depth == 0)
            continue;

        Image3D zero(visibleBox.dimensions, 1.0);
        image.subtractSubimage(visibleBox, zero);

    }
    return image.isEqual(0.0);
}

void expectViableSplit(const CuboidShape& imageSize, const Result<std::vector<BoxCoordWithPadding>>& result) {
    ASSERT_TRUE(result.success);
    ASSERT_FALSE(result.value.empty());
    EXPECT_TRUE(checkSameShape(imageSize, result.value))
        << "not all cubes share the same padded shape";
    EXPECT_TRUE(checkCompleteness(imageSize, result.value))
        << "cubes do not cover the complete image";
}

std::string printShape(const CuboidShape& imageSize,
                       const Padding& cubePadding,
                       const Padding& imagePadding,
                       const size_t maxVolumePerCube,
                       const size_t minNumberCubes,
                       const CuboidShape& minShape,
                       const Result<std::vector<BoxCoordWithPadding>>& result) {
    std::string s = "imageSize: " + imageSize.print()
        + ", cubePadding: before " + cubePadding.before.print() + " / after " + cubePadding.after.print()
        + ", imagePadding: before " + imagePadding.before.print() + " / after " + imagePadding.after.print()
        + ", maxVolumePerCube: " + std::to_string(maxVolumePerCube)
        + ", minNumberCubes: " + std::to_string(minNumberCubes)
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
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(1, 1, 1));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 1000000, 1, CuboidShape(1, 1, 1), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ImageWithPadding) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 1000000, 1, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 1000000, 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, LargeImageMultipleCubes) {
    CuboidShape imageSize(512, 512, 100);
    Padding padding{CuboidShape(32, 32, 16), CuboidShape(32, 32, 16)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 1000000, 4, CuboidShape(65, 65, 33));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 1000000, 4, CuboidShape(65, 65, 33), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 4u);
}


TEST_F(ImageSplitTest, VeryLargeImage) {
    CuboidShape imageSize(2000, 13000, 200);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 9e9, 8, CuboidShape(9, 9, 5));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 9e9, 8, CuboidShape(9, 9, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, VerySmallImage) {
    CuboidShape imageSize(32, 32, 10);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(9, 9, 5));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 1000000, 1, CuboidShape(9, 9, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, NonePaddingType) {
    CuboidShape imageSize(200, 200, 80);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 500000, 2, CuboidShape(41, 41, 21));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 500000, 2, CuboidShape(41, 41, 21), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, CubeCoverageVerification) {
    CuboidShape imageSize(128, 128, 32);
    Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500000, 1, CuboidShape(17, 17, 9));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 500000, 1, CuboidShape(17, 17, 9), result);
    expectViableSplit(imageSize, result);
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
    auto result = splitImageHomogeneous(padding, padding, imageSize, 1000000, 8, CuboidShape(33, 33, 17));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 1000000, 8, CuboidShape(33, 33, 17), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 8u);
}

TEST_F(ImageSplitTest, ConstrainedMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 50000, 1, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 50000, 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, AsymmetricImage) {
    CuboidShape imageSize(400, 200, 30);
    Padding padding{CuboidShape(20, 10, 5), CuboidShape(20, 10, 5)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 300000, 2, CuboidShape(41, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 300000, 2, CuboidShape(41, 21, 11), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, FailTooLittleMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500, 1, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 500, 1, CuboidShape(21, 21, 11), result);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageSplitTest, FailTooManyCubes) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 5000, 100, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 5000, 100, CuboidShape(21, 21, 11), result);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageSplitTest, FirstCubeAtOriginNonePadding) {
    CuboidShape imageSize(128, 128, 32);
    Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 500000, 1, CuboidShape(17, 17, 9));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 500000, 1, CuboidShape(17, 17, 9), result);
    expectViableSplit(imageSize, result);

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
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(5, 5, 5));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 1000000, 1, CuboidShape(5, 5, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, NoPadding) {
    CuboidShape imageSize(64, 64, 32);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 500000, 1, CuboidShape(1, 1, 1));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 500000, 1, CuboidShape(1, 1, 1), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, LargePaddingRelative) {
    CuboidShape imageSize(50, 50, 20);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500000, 1, CuboidShape(41, 41, 21));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 500000, 1, CuboidShape(41, 41, 21), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, CubeSizeEqualsImage) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitImageHomogeneous(padding, padding, imageSize, 10000000, 1, CuboidShape(21, 21, 11));
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 10000000, 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}
