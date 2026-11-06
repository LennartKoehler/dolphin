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
#include "dolphinbackend/MemoryFootprint.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin/Logging.h"
#include <algorithm>

// ----------------------------------------------------------------------------
// Test footprints
//
// The splitter validates each candidate (padded) cube shape against a
// MemoryBudget by asking a MemoryFootprint. These components model the
// footprint shapes the tests need:
//  - VolumeCopiesComponent: N copies of the cube volume per pool (the common
//    "everything scales with the cube" case)
//  - FixedOverheadComponent: constant bytes regardless of the shape
//  - TileQuantizedComponent: cost constant per z-band (models strip/tile
//    readers: piecewise constant, plateau while shrinking z)
// ----------------------------------------------------------------------------

class VolumeCopiesComponent : public MemoryComponent {
public:
    VolumeCopiesComponent(size_t hostCopies, size_t deviceCopies)
        : hostCopies(hostCopies), deviceCopies(deviceCopies) {}

    MemoryRequirement footprint(const CuboidShape& paddedCube) const override {
        const size_t bytes = paddedCube.getVolume() * sizeof(real_t);
        return MemoryRequirement{hostCopies * bytes, deviceCopies * bytes};
    }

    std::string name() const override { return "volume copies"; }

private:
    size_t hostCopies;
    size_t deviceCopies;
};

class FixedOverheadComponent : public MemoryComponent {
public:
    explicit FixedOverheadComponent(size_t hostBytes) : hostBytes(hostBytes) {}

    MemoryRequirement footprint(const CuboidShape&) const override {
        return MemoryRequirement{hostBytes, 0};
    }

    std::string name() const override { return "fixed overhead"; }

private:
    size_t hostBytes;
};

// Cost of reading a region from a banded file: whole z-bands of the full
// padded width/height must be held, so the cost only drops when the candidate
// crosses a band boundary (plateaus while shrinking z).
class TileQuantizedComponent : public MemoryComponent {
public:
    explicit TileQuantizedComponent(size_t bandDepth) : bandDepth(bandDepth) {}

    MemoryRequirement footprint(const CuboidShape& paddedCube) const override {
        const size_t bands = (paddedCube.depth + bandDepth - 1) / bandDepth;
        const size_t bytes = bands * bandDepth * paddedCube.width * paddedCube.height * sizeof(real_t);
        return MemoryRequirement{bytes, 0};
    }

    std::string name() const override { return "tile quantized"; }

private:
    size_t bandDepth;
};

static MemoryBudget volumeBudgetBytes(size_t hostBytes) {
    MemoryBudget budget;
    budget.deviceIsHost = true;
    budget.hostTotal_byte = hostBytes;
    budget.deviceTotal_byte = budget.hostTotal_byte;
    return budget;
}

// Budget in real_t elements for a merged host pool (CPU backend: device ==
// host), with a single volume-copies footprint of 1 copy. Accepts a candidate
// iff its padded volume <= maxPaddedVolumeElements.
static MemoryBudget volumeBudget(size_t maxPaddedVolumeElements) {
    return volumeBudgetBytes(maxPaddedVolumeElements * sizeof(real_t));
}

static Result<std::vector<BoxCoordWithPadding>> splitVolumeLimited(
    const Padding& cubePadding,
    const Padding& imagePadding,
    const CuboidShape& imageOriginalShape,
    const size_t& maxPaddedVolumeElements,
    const size_t& minNumberCubes,
    const CuboidShape& minSize,
    bool sliceAllDimensions
){
    VolumeCopiesComponent component(1, 0);
    MemoryFootprint footprint;
    footprint.add(component);
    return splitImageHomogeneous(
        cubePadding, imagePadding, imageOriginalShape,
        volumeBudget(maxPaddedVolumeElements), footprint,
        minNumberCubes, minSize, sliceAllDimensions);
}

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
                       const MemoryBudget& budget,
                       const size_t minNumberCubes,
                       const CuboidShape& minShape,
                       const Result<std::vector<BoxCoordWithPadding>>& result) {
    std::string s = "imageSize: " + imageSize.print()
        + ", cubePadding: before " + cubePadding.before.print() + " / after " + cubePadding.after.print()
        + ", imagePadding: before " + imagePadding.before.print() + " / after " + imagePadding.after.print()
        + ", budget: " + std::to_string(budget.hostTotal_byte) + " B host / "
        + std::to_string(budget.deviceTotal_byte) + " B device"
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
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(1, 1, 1), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(1000000), 1, CuboidShape(1, 1, 1), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ImageWithPadding) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 1000000, 1, CuboidShape(21, 21, 11), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(1000000), 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, LargeImageMultipleCubes) {
    CuboidShape imageSize(256, 256, 50);
    Padding padding{CuboidShape(32, 32, 16), CuboidShape(32, 32, 16)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 1000000, 4, CuboidShape(65, 65, 33), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(1000000), 4, CuboidShape(65, 65, 33), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 4u);
}

TEST_F(ImageSplitTest, VeryLargeImage) {
    CuboidShape imageSize(2000, 13000, 200);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 9000000000ull, 8, CuboidShape(9, 9, 5), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(9000000000ull), 8, CuboidShape(9, 9, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, VerySmallImage) {
    CuboidShape imageSize(32, 32, 10);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(9, 9, 5), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(1000000), 1, CuboidShape(9, 9, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, NonePaddingType) {
    CuboidShape imageSize(200, 200, 80);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 500000, 2, CuboidShape(41, 41, 21), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(500000), 2, CuboidShape(41, 41, 21), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, CubeCoverageVerification) {
    CuboidShape imageSize(128, 128, 32);
    Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 500000, 1, CuboidShape(17, 17, 9), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(500000), 1, CuboidShape(17, 17, 9), result);
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
    auto result = splitVolumeLimited(padding, padding, imageSize, 1000000, 8, CuboidShape(33, 33, 17), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(1000000), 8, CuboidShape(33, 33, 17), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 8u);
}

TEST_F(ImageSplitTest, ConstrainedMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 500000, 1, CuboidShape(21, 21, 11), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(500000), 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, AsymmetricImage) {
    CuboidShape imageSize(400, 200, 30);
    Padding padding{CuboidShape(20, 10, 5), CuboidShape(20, 10, 5)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 300000, 2, CuboidShape(41, 21, 11), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(300000), 2, CuboidShape(41, 21, 11), result);
    expectViableSplit(imageSize, result);
    EXPECT_GE(result.value.size(), 2u);
}

TEST_F(ImageSplitTest, FailTooLittleMemory) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 500, 1, CuboidShape(21, 21, 11), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(500), 1, CuboidShape(21, 21, 11), result);
    EXPECT_FALSE(result.success);
}

TEST_F(ImageSplitTest, MinNumberCubesIsFloor) {
    CuboidShape imageSize(200, 200, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 200000, 20, CuboidShape(21, 21, 11), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(200000), 20, CuboidShape(21, 21, 11), result);
    // minNumberCubes is a lower bound (e.g. to fill all devices), not a cap:
    // a large but satisfiable request succeeds by producing enough cubes.
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 20u);
}

TEST_F(ImageSplitTest, MinimumViableImage) {
    CuboidShape imageSize(10, 10, 10);
    Padding padding{CuboidShape(2, 2, 2), CuboidShape(2, 2, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 1000000, 1, CuboidShape(5, 5, 5), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(1000000), 1, CuboidShape(5, 5, 5), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ImageSplitFixedSize) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(4, 4, 4), CuboidShape(4, 4, 4)};
    CuboidShape fixedSize{45, 45, 45};
    int minNumberCubes = 1;
    auto result = splitImageHomogeneousFixedSize(padding, imagePadding, imageSize, fixedSize);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, MemoryBudget{}, minNumberCubes, CuboidShape(5, 5, 5), result);
    expectViableSplit(imageSize, result);
    checkSameShapeReference(fixedSize, result.value);
}

TEST_F(ImageSplitTest, NoPadding) {
    CuboidShape imageSize(64, 64, 32);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, 500000, 1, CuboidShape(1, 1, 1), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(500000), 1, CuboidShape(1, 1, 1), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, LargePaddingRelative) {
    CuboidShape imageSize(50, 50, 20);
    Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 500000, 1, CuboidShape(41, 41, 21), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(500000), 1, CuboidShape(41, 41, 21), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, CubeSizeEqualsImage) {
    CuboidShape imageSize(100, 100, 50);
    Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
    auto result = splitVolumeLimited(padding, padding, imageSize, 10000000, 1, CuboidShape(21, 21, 11), true);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, padding, volumeBudget(10000000), 1, CuboidShape(21, 21, 11), result);
    expectViableSplit(imageSize, result);
}

TEST_F(ImageSplitTest, ZOnly_ShrinksZNotXY) {
    CuboidShape imageSize(100, 100, 100);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(31, 31, 21);
    size_t maxVolume = 1000000;

    // Z-only: memory pressure must be absorbed by Z, X/Y stay at full (padded) size
    auto zOnly = splitVolumeLimited(padding, imagePadding, imageSize, maxVolume, 1, minSize, false);
    ASSERT_TRUE(zOnly.success) << printShape(imageSize, padding, imagePadding, volumeBudget(maxVolume), 1, minSize, zOnly);
    expectViableSplit(imageSize, zOnly);
    CuboidShape zOnlyPadded = zOnly.value[0].getPaddedShape();
    EXPECT_EQ(zOnlyPadded.width, 100u);
    EXPECT_EQ(zOnlyPadded.height, 100u);
    EXPECT_LT(zOnlyPadded.depth, 120u);

    // All dimensions: same params, X must shrink too (proves the modes behave differently)
    auto allDims = splitVolumeLimited(padding, imagePadding, imageSize, maxVolume, 1, minSize, true);
    ASSERT_TRUE(allDims.success) << printShape(imageSize, padding, imagePadding, volumeBudget(maxVolume), 1, minSize, allDims);
    EXPECT_LT(allDims.value[0].getPaddedShape().width, 1000u);
}

TEST_F(ImageSplitTest, ZOnly_RespectsMinCubes) {
    CuboidShape imageSize(1000, 1000, 400);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(31, 31, 21);
    size_t maxVolume = 1000000000;

    auto result = splitVolumeLimited(padding, imagePadding, imageSize, maxVolume, 8, minSize, false);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(maxVolume), 8, minSize, result);
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
    auto zOnly = splitVolumeLimited(padding, imagePadding, imageSize, maxVolume, 1, minSize, false);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(maxVolume), 1, minSize, zOnly);
    ASSERT_FALSE(zOnly.success);

    // All dimensions: X/Y can still shrink, so the same params succeed
    auto allDims = splitVolumeLimited(padding, imagePadding, imageSize, maxVolume, 1, minSize, true);
    EXPECT_TRUE(allDims.success) << printShape(imageSize, padding, imagePadding, volumeBudget(maxVolume), 1, minSize, allDims);
}

TEST_F(ImageSplitTest, ZOnly_ReducePassKeepsXY) {
    CuboidShape imageSize(100, 100, 100);
    Padding padding{CuboidShape(10, 10, 10), CuboidShape(10, 10, 10)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    CuboidShape minSize(31, 31, 21);
    size_t maxVolume = imageSize.getVolume() / 2.5;

    // Main loop picks the largest fitting z; the follow-up reduce pass must
    // not shrink X/Y to keep the cube count (Z-only slicing): X/Y stay at the
    // full image size and only the cube count in z adapts.
    auto result = splitVolumeLimited(padding, imagePadding, imageSize, maxVolume, 1, minSize, false);
    GTEST_LOG_(INFO) << printShape(imageSize, padding, imagePadding, volumeBudget(maxVolume), 1, minSize, result);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.value.size(), 2u);
    expectViableSplit(imageSize, result);
    CuboidShape padded = result.value[0].getPaddedShape();
    EXPECT_EQ(padded.width, 100u);
    EXPECT_EQ(padded.height, 100u);
    EXPECT_GE(padded.depth, 33u);
}

// ---------------------------------------------------------------------------
// Footprint-driven tests: different MemoryComponent shapes bound the split.
// ---------------------------------------------------------------------------

// Multiple components add up before the budget check: the cube must fit the
// SUM of all components, not each individually.
TEST_F(ImageSplitTest, MultiComponentFootprint_SumsBeforeComparing) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    // reader 2 copies + io staging 3 copies + algorithm 1 copy = 6 copies total
    VolumeCopiesComponent reader(2, 0);
    VolumeCopiesComponent io(3, 0);
    VolumeCopiesComponent algorithm(1, 0);
    MemoryFootprint footprint;
    footprint.add(reader);
    footprint.add(io);
    footprint.add(algorithm);

    // 32^3 = 32768 elements per cube -> 6 copies must fit exactly
    const size_t maxCubeElements = 32u * 32u * 32u;
    MemoryBudget budget = volumeBudget(maxCubeElements * 6);

    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(1, 1, 1), true);
    ASSERT_TRUE(result.success);
    // the chosen padded cube must satisfy the full footprint
    const CuboidShape padded = result.value[0].getPaddedShape();
    EXPECT_LE(footprint.total(padded).hostBytes, budget.hostTotal_byte);
    // ... and must not be larger than the single-copy budget allows
    EXPECT_LE(padded.getVolume(), maxCubeElements);
}

// With separate pools the device budget can bind while the host has slack:
// growing the device budget must grow the chosen cubes.
TEST_F(ImageSplitTest, DevicePoolBindsWhenHostHasSlack) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    // algorithm lives on the device; staging on the host
    VolumeCopiesComponent staging(3, 0);
    VolumeCopiesComponent algorithm(0, 1);
    MemoryFootprint footprint;
    footprint.add(staging);
    footprint.add(algorithm);

    MemoryBudget budget;
    budget.deviceIsHost = false;
    budget.hostTotal_byte = 1ull << 30;                      // host: plenty
    budget.deviceTotal_byte = 512u * 1024;                   // device: 512 KiB -> 128K elements

    auto small = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(1, 1, 1), true);
    ASSERT_TRUE(small.success);
    const size_t smallVolume = small.value[0].getPaddedShape().getVolume();
    EXPECT_LE(smallVolume * sizeof(real_t), budget.deviceTotal_byte);

    // 2x the device budget -> the whole image fits on the device in one cube
    budget.deviceTotal_byte = 1024u * 1024;
    auto large = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(1, 1, 1), true);
    ASSERT_TRUE(large.success);
    const size_t largeVolume = large.value[0].getPaddedShape().getVolume();
    EXPECT_GT(largeVolume, smallVolume);
    EXPECT_LE(largeVolume * sizeof(real_t), budget.deviceTotal_byte);
}

// CPU backend: device == host, so host and device demands draw from ONE pool
// and their sum must fit.
TEST_F(ImageSplitTest, CpuBackendMergesBothPools) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    // 2 copies on the "host plane" + 3 copies on the "device plane" = 5 total
    VolumeCopiesComponent hostPart(2, 0);
    VolumeCopiesComponent devicePart(0, 3);
    MemoryFootprint footprint;
    footprint.add(hostPart);
    footprint.add(devicePart);

    const size_t cubeElements = 16u * 16u * 16u; // 4096
    const size_t totalBytes = 5 * cubeElements * sizeof(real_t);

    MemoryBudget fits = volumeBudgetBytes(totalBytes);
    auto ok = splitImageHomogeneous(padding, imagePadding, imageSize, fits, footprint, 1, CuboidShape(1, 1, 1), true);
    ASSERT_TRUE(ok.success);
    EXPECT_LE(ok.value[0].getPaddedShape().getVolume(), cubeElements);

    // One byte less than the minimum cube's summed demand: host 2*4 B +
    // device 3*4 B = 20 B for a 1x1x1 cube. Either plane alone would fit,
    // only the merged-pool accounting rejects it.
    MemoryBudget tooSmall = volumeBudgetBytes(19);
    auto fail = splitImageHomogeneous(padding, imagePadding, imageSize, tooSmall, footprint, 1, CuboidShape(1, 1, 1), true);
    EXPECT_FALSE(fail.success);
}

// A component with a constant cost (independent of the cube) shrinks the
// remaining budget for the volume-scaled parts.
TEST_F(ImageSplitTest, FixedOverheadShrinksRemainingBudget) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    FixedOverheadComponent overhead(900u * 1024); // 900 KiB, shape-independent
    VolumeCopiesComponent volume(1, 0);
    MemoryFootprint footprint;
    footprint.add(overhead);
    footprint.add(volume);

    // 1 MiB budget -> 124 KiB left for the cube -> 31744 elements
    const size_t budgetBytes = 1024u * 1024;
    MemoryBudget budget = volumeBudgetBytes(budgetBytes);

    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(1, 1, 1), true);
    ASSERT_TRUE(result.success);
    const CuboidShape padded = result.value[0].getPaddedShape();
    EXPECT_LE(padded.getVolume() * sizeof(real_t), budgetBytes - 900u * 1024);

    // Budget below the fixed overhead plus the minimum cube: nothing fits.
    MemoryBudget tooSmall = volumeBudgetBytes(900u * 1024 + 3);
    auto fail = splitImageHomogeneous(padding, imagePadding, imageSize, tooSmall, footprint, 1, CuboidShape(1, 1, 1), true);
    EXPECT_FALSE(fail.success);
    EXPECT_NE(fail.getErrorString().find("fixed overhead"), std::string::npos)
        << "failure must name the binding component: " << fail.getErrorString();
}

// Tile/strip readers have piecewise-constant costs: shrinking z within a band
// does not reduce the demand (plateau), but the split must still find a fit.
TEST_F(ImageSplitTest, TileQuantizedCost_PlateauStillFits) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    TileQuantizedComponent reader(8); // 8-slice bands, full width x height
    MemoryFootprint footprint;
    footprint.add(reader);

    // one band of 64x64x8 = 128 KiB; two bands do not fit
    MemoryBudget budget = volumeBudgetBytes(128u * 1024);

    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(1, 1, 1), false);
    ASSERT_TRUE(result.success) << result.getErrorString();
    // the chosen cube must stay within one band
    EXPECT_LE(result.value[0].getPaddedShape().depth, 8u);
    expectViableSplit(imageSize, result);
}

// Failure must explain which components bound the budget.
TEST_F(ImageSplitTest, FailureReportsFootprintBreakdown) {
    CuboidShape imageSize(64, 64, 64);
    Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    FixedOverheadComponent overhead(4096);
    VolumeCopiesComponent volume(2, 0);
    MemoryFootprint footprint;
    footprint.add(overhead);
    footprint.add(volume);

    MemoryBudget budget = volumeBudgetBytes(0);
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(1, 1, 1), true);
    ASSERT_FALSE(result.success);
    EXPECT_NE(result.getErrorString().find("no cube size fits"), std::string::npos)
        << result.getErrorString();
    EXPECT_NE(result.getErrorString().find("fixed overhead"), std::string::npos)
        << result.getErrorString();
    EXPECT_NE(result.getErrorString().find("volume copies"), std::string::npos)
        << result.getErrorString();
}

// A footprint that demands nothing always admits the largest cube.
TEST_F(ImageSplitTest, EmptyFootprint_SingleCube) {
    CuboidShape imageSize(64, 64, 32);
    Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
    Padding imagePadding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};

    VolumeCopiesComponent nothing(0, 0);
    MemoryFootprint footprint;
    footprint.add(nothing);

    MemoryBudget budget = volumeBudgetBytes(1);
    auto result = splitImageHomogeneous(padding, imagePadding, imageSize, budget, footprint, 1, CuboidShape(10, 10, 10), true);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.value.size(), 1u);
    expectViableSplit(imageSize, result);
}

// ---------------------------------------------------------------------------
// Geometric contract (the properties the splitter must hold for ANY shape)
//
// For any image shape N, paddings (inside >= outside) and padded cube shape T
// (with T >= inside.total per dim), every split must satisfy:
//   (1) uniform : every cube's padded shape == T
//   (2) bounds  : every core lies inside the image, core size >= 1
//   (3) tiling  : cores cover the image EXACTLY once (no gap, no overlap)
//   (4) faces   : face at image start: before >= outside.before
//                 face at image end:   after  == outside.after
//                 interior faces:      before/after >= inside
struct Contract {
    CuboidShape image, T;
    Padding inside, outside;
    std::vector<BoxCoordWithPadding> cubes;
};

// Returns the number of cubes in the split (so callers can track multi-cube coverage).
size_t expectContract(const Contract& c) {
    if (c.cubes.empty()) return 0;
    CuboidShape T = c.T;
    for (const auto& cube : c.cubes) {
        // (1) uniform padded shape
        EXPECT_EQ(cube.getPaddedShape(), T)
            << "cube " << cube.print() << " padded shape != T " << T.print();
        // (2) bounds
        for (size_t d = 0; d < 3; ++d) {
            int64_t p = cube.box.position.at(d);
            size_t core = cube.box.dimensions.at(d);
            int64_t N = static_cast<int64_t>(c.image.at(d));
            EXPECT_GE(p, 0) << "core starts before image, dim " << d;
            EXPECT_LE(p + static_cast<int64_t>(core), N)
                << "core overshoots image, dim " << d << " (p=" << p << " core=" << core << " N=" << N << ")";
            EXPECT_GT(core, 0u) << "empty core, dim " << d;
            // (4) faces
            if (p == 0)
                EXPECT_GE(cube.padding.before.at(d), c.outside.before.at(d))
                    << "boundary before-face " << cube.padding.before.at(d)
                    << " < outsidePadding " << c.outside.before.at(d) << " dim " << d;
            else
                EXPECT_GE(cube.padding.before.at(d), c.inside.before.at(d))
                    << "interior before-face " << cube.padding.before.at(d)
                    << " < insidePadding " << c.inside.before.at(d) << " dim " << d;
            if (p + static_cast<int64_t>(core) == N)
                EXPECT_EQ(cube.padding.after.at(d), c.outside.after.at(d))
                    << "boundary after-face " << cube.padding.after.at(d)
                    << " != outsidePadding " << c.outside.after.at(d) << " dim " << d;
            else
                EXPECT_GE(cube.padding.after.at(d), c.inside.after.at(d))
                    << "interior after-face " << cube.padding.after.at(d)
                    << " < insidePadding " << c.inside.after.at(d) << " dim " << d;
        }
    }
    // (3) exact-once tiling
    if (c.image.getVolume() <= maxCompletenessCheckVolume) {
        Image3D counter(c.image, 1.0f); // 1 = untouched
        for (const auto& cube : c.cubes) {
            Image3D one(cube.box.dimensions, 1.0f);
            counter.subtractSubimage(cube.box, one); // covered once -> 0, twice -> -1
        }
        EXPECT_TRUE(counter.isEqual(0.0f))
            << "cores do not tile the image exactly once";
    }
    return c.cubes.size();
}


// Proves the exact-once tiling detector is not vacuous: a deliberately broken
// split (a 2px gap between two cores) must register as "not exactly once".
TEST_F(ImageSplitTest, Contract_CheckerDetectsGap) {
    CuboidShape image(10, 4, 4);
    // Two cores [0,4) and [6,10) leave x in [4,6) uncovered.
    BoxCoord gapA{{0, 0, 0}, {4, 4, 4}};
    BoxCoord gapB{{6, 0, 0}, {4, 4, 4}};
    Image3D counter(image, 1.0f);
    counter.subtractSubimage(gapA, Image3D(gapA.dimensions, 1.0f));
    counter.subtractSubimage(gapB, Image3D(gapB.dimensions, 1.0f));
    EXPECT_FALSE(counter.isEqual(0.0f))
        << "tiling detector must flag a split with a gap";

    // Sanity: the same detector accepts a correct tiling.
    BoxCoord okA{{0, 0, 0}, {5, 4, 4}};
    BoxCoord okB{{5, 0, 0}, {5, 4, 4}};
    Image3D counterOk(image, 1.0f);
    counterOk.subtractSubimage(okA, Image3D(okA.dimensions, 1.0f));
    counterOk.subtractSubimage(okB, Image3D(okB.dimensions, 1.0f));
    EXPECT_TRUE(counterOk.isEqual(0.0f))
        << "tiling detector must accept an exact tiling";
}
