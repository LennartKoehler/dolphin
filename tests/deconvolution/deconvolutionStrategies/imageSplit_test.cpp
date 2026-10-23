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
bool checkSameShape(const std::vector<BoxCoordWithPadding>& cubes){
    if (cubes.empty()) return false;
    BoxCoordWithPadding previous = cubes[0];
    for (const BoxCoordWithPadding& cube : cubes){
       if (cube.getPaddedBox().dimensions != previous.getPaddedBox().dimensions){
            GTEST_LOG_(WARNING) << "Previous BoxCoordWithPadding: " << previous.print() << "; Current BoxCoordWithPadding: " << cube.print();
            return false;
        }
    }
    return true;
}

bool checkSameShapeReference(CuboidShape reference, const std::vector<BoxCoordWithPadding>& cubes){
    if (cubes.empty()) return false;
    for (const BoxCoordWithPadding& cube : cubes){
       if (cube.getPaddedBox().dimensions != reference){
            GTEST_LOG_(WARNING) << "reference BoxCoordWithPadding: " << reference.print() << "; Current BoxCoordWithPadding: " << cube.print();
            return false;
        }
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
    EXPECT_TRUE(checkSameShape(result.value))
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
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(1, 1, 1), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 1000000, 1, CuboidShape(1, 1, 1), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ImageWithPadding) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 1000000, 1, CuboidShape(21, 21, 11), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 1000000, 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, LargeImageMultipleCubes) {
    CuboidShape imageSize(256, 256, 50);
    Padding padding{CuboidShape(32, 32, 16), CuboidShape(32, 32, 16)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 1000000, 4, CuboidShape(65, 65, 33), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 1000000, 4, CuboidShape(65, 65, 33), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 4u);
}

TEST_F(ImageSplitTest, VeryLargeImage) {
    CuboidShape imageSize(2000, 13000, 200);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 9e9, 8, CuboidShape(9, 9, 5), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 9e9, 8, CuboidShape(9, 9, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, VerySmallImage) {
    CuboidShape imageSize(32, 32, 10);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(9, 9, 5), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 1000000, 1, CuboidShape(9, 9, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, NonePaddingType) {
    CuboidShape imageSize(200, 200, 80);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 500000, 2, CuboidShape(41, 41, 21), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 500000, 2, CuboidShape(41, 41, 21), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, CubeCoverageVerification) {
    CuboidShape imageSize(128, 128, 32);
    Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500000, 1, CuboidShape(17, 17, 9), sliceAllDimensions);
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
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 1000000, 8, CuboidShape(33, 33, 17), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 1000000, 8, CuboidShape(33, 33, 17), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 8u);
}

TEST_F(ImageSplitTest, ConstrainedMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500000, 1, CuboidShape(21, 21, 11), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 50000, 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, AsymmetricImage) {
    CuboidShape imageSize(400, 200, 30);
    Padding padding{CuboidShape(20, 10, 5), CuboidShape(20, 10, 5)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 300000, 2, CuboidShape(41, 21, 11), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 300000, 2, CuboidShape(41, 21, 11), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, FailTooLittleMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500, 1, CuboidShape(21, 21, 11), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 500, 1, CuboidShape(21, 21, 11), result);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageSplitTest, MinNumberCubesIsFloor) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 200000, 20, CuboidShape(21, 21, 11), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 5000, 20, CuboidShape(21, 21, 11), result);
    // minNumberCubes is a lower bound (e.g. to fill all devices), not a cap:
    // a large but satisfiable request succeeds by producing enough cubes.
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 20u);
}

// if only one cube in that dimensions then imagesplit might add padding before because the last cube (which is the first cube) is "shifted backward" (padding added before)
//
// TEST_F(ImageSplitTest, FirstCubeAtOriginNonePadding) {
//     CuboidShape imageSize(128, 128, 32);
//     Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
//     Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
//     bool sliceAllDimensions = true;
//     auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 500000, 2, CuboidShape(17, 17, 9), sliceAllDimensions);
//     GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 500000, 2, CuboidShape(17, 17, 9), result);
//     expectViableSplit(imageSize, result);
//
//     bool foundOriginCube = false;
//     for (const auto& cube : result.value) {
//         if (cube.box.position.width == 0 && cube.box.position.height == 0 && cube.box.position.depth == 0) {
//             foundOriginCube = true;
//             EXPECT_EQ(cube.padding.before.width, 0);
//             EXPECT_EQ(cube.padding.before.height, 0);
//             EXPECT_EQ(cube.padding.before.depth, 0);
//             break;
//         }
//     }
//     EXPECT_TRUE(foundOriginCube);
// }

TEST_F(ImageSplitTest, MinimumViableImage) {
    CuboidShape imageSize(10, 10, 10);
    Padding padding{CuboidShape(2, 2, 2), CuboidShape(2, 2, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(5, 5, 5), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 1000000, 1, CuboidShape(5, 5, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ImageSplitFixedSize) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(4, 4, 4), CuboidShape(4, 4, 4)};
    CuboidShape fixedSize{45, 45, 45};
    bool sliceAllDimensions = true;
    CuboidShape minSize = CuboidShape(5, 5, 5);
    size_t maxMemoryPerCube = 1000000;
    int minNumberCubes = 1;
    auto result = splitImageHomogeneousFixedSize(padding, imagePadding, imageSize, fixedSize);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, maxMemoryPerCube, minNumberCubes, minSize, result);
    expectViableSplit(imageSize, result);
    checkSameShapeReference(fixedSize, result.value);
}

TEST_F(ImageSplitTest, NoPadding) {
    CuboidShape imageSize(64, 64, 32);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, 500000, 1, CuboidShape(1, 1, 1), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, 500000, 1, CuboidShape(1, 1, 1), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, LargePaddingRelative) {
    CuboidShape imageSize(50, 50, 20);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 500000, 1, CuboidShape(41, 41, 21), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 500000, 1, CuboidShape(41, 41, 21), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, CubeSizeEqualsImage) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    bool sliceAllDimensions = true;
    auto result = splitImageHomogeneous(padding, padding, imageSize, 10000000, 1, CuboidShape(21, 21, 11), sliceAllDimensions);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, 10000000, 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ZOnly_ShrinksZNotXY) {
    CuboidShape imageSize(100, 100, 100);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(31, 31, 21);
    size_t maxVolume = 1000000;

    // Z-only: memory pressure must be absorbed by Z, X/Y stay at full (padded) size
    auto zOnly = splitImageHomogeneous(padding, imagePadding, imageSize, maxVolume, 1, minSize, false);
    ASSERT_TRUE(zOnly.success) << printShape(imageSize, padding, imagePadding, maxVolume, 1, minSize, zOnly);
    expectViableSplit(imageSize, zOnly);
    CuboidShape zOnlyPadded = zOnly.value[0].getPaddedShape();
    EXPECT_EQ(zOnlyPadded.width, 100u);
    EXPECT_EQ(zOnlyPadded.height, 100u);
    EXPECT_LT(zOnlyPadded.depth, 120u);

    // All dimensions: same params, X must shrink too (proves the modes behave differently)
    auto allDims = splitImageHomogeneous(padding, imagePadding, imageSize, maxVolume, 1, minSize, true);
    ASSERT_TRUE(allDims.success) << printShape(imageSize, padding, imagePadding, maxVolume, 1, minSize, allDims);
    EXPECT_LT(allDims.value[0].getPaddedShape().width, 1000u);
}

TEST_F(ImageSplitTest, ZOnly_RespectsMinCubes) {
    CuboidShape imageSize(1000, 1000, 400);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(31, 31, 21);
    size_t maxVolume = 1000000000;

    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, maxVolume, 8, minSize, false);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, maxVolume, 8, minSize, result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 8u);
    expectViableSplit(imageSize, result);
    CuboidShape padded = result.value[0].getPaddedShape();
    EXPECT_EQ(padded.width, 1000u);
    EXPECT_EQ(padded.height, 1000u);
    EXPECT_LT(padded.depth, 400u);
}

TEST_F(ImageSplitTest, ZOnly_FailsWhenZAtMin) {
    CuboidShape imageSize(100, 100, 30);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(21, 21, 21);
    size_t maxVolume = 80000;

    // Z-only: Z cannot shrink below minSize, so it must fail
    auto zOnly = splitImageHomogeneous(padding, imagePadding, imageSize, maxVolume, 1, minSize, false);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, maxVolume, 1, minSize, zOnly);
    ASSERT_FALSE(zOnly.success);

    // All dimensions: X/Y can still shrink, so the same params succeed
    auto allDims = splitImageHomogeneous(padding, imagePadding, imageSize, maxVolume, 1, minSize, true);
    EXPECT_TRUE(allDims.success) << printShape(imageSize, padding, imagePadding, maxVolume, 1, minSize, allDims);
}

TEST_F(ImageSplitTest, ZOnly_ReducePassKeepsXY) {
    CuboidShape imageSize(100, 100, 100);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(31, 31, 21);
    size_t maxVolume = imageSize.getVolume() / 2.5;

    // Main loop picks maxSize 3000x3000x24; the follow-up reduce pass must not
    // shrink X/Y to keep the cube count (Z-only slicing), otherwise width drops
    // well below 3000 (e.g. to 1200) while Z already provides the slabs.
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, maxVolume, 1, minSize, false);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, maxVolume, 1, minSize, result);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.value.size(), 5u);
    expectViableSplit(imageSize, result);
    CuboidShape padded = result.value[0].getPaddedShape();
    EXPECT_EQ(padded.width, 100u);
    EXPECT_EQ(padded.height, 100u);
    EXPECT_GE(padded.depth, 33u);
}
