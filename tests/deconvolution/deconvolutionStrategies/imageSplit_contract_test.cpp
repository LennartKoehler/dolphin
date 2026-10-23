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

// Exhaustive geometric-contract tests for image splitting.
// Labeled "exhaustive" in ctest — excluded from regular CI runs.
// Run them with:  ctest -L exhaustive --output-on-failure
//                 (or: ./imageSplit_contract_test)

#include <gtest/gtest.h>
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin/Logging.h"

class ImageSplitContractTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

constexpr size_t maxCompletenessCheckVolume = 100000000; // ~400 MB as float image

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

// Brute force: fixed-T splits over many image shapes / paddings / T.
TEST_F(ImageSplitContractTest, FixedT_AnyShape) {
    size_t checked = 0, multiCube = 0;
    for (size_t nx : {8u, 12u, 20u, 33u, 48u, 64u})
        for (size_t ny : {5u, 15u, 24u, 40u})
            for (size_t nz : {3u, 9u, 16u, 28u})
                for (size_t pin : {0u, 1u, 2u, 4u, 8u})
                    for (size_t pout : {static_cast<size_t>(0), pin})
                        for (size_t t : {1u, 2u, 3u, 4u, 5u, 6u, 8u, 10u, 12u, 16u, 20u, 24u, 32u, 40u, 48u, 64u}) {
                            if (t <= 2 * pin) continue; // T must fit interior padding + >=1 core
                            CuboidShape image(nx, ny, nz);
                            Padding inside{CuboidShape(pin, pin, pin), CuboidShape(pin, pin, pin)};
                            Padding outside{CuboidShape(pout, pout, pout), CuboidShape(pout, pout, pout)};
                            CuboidShape T(t, t, t);
                            auto r = splitImageHomogeneousFixedSize(inside, outside, image, T);
                            if (!r.success) continue;
                            ++checked;
                            if (r.value.size() > 1) ++multiCube;
                            expectContract({image, T, inside, outside, r.value});
                        }
    GTEST_LOG_(INFO) << "FixedT_AnyShape: " << checked << " splits verified, "
        << multiCube << " multi-cube";
    EXPECT_GE(checked, 1000u);
    EXPECT_GE(multiCube, 100u) << "sweep must exercise multi-cube splits, not just single-cube";
}

// Same contract for the homogeneous (auto-sized) path.
// splitImageHomogeneous's size-reduction loop is expensive for very small
// maxVolume / large images, so use a bounded, targeted list of cases that
// includes both single- and multi-cube splits across paddings and slice modes.
TEST_F(ImageSplitContractTest, Homogeneous_AnyShape) {
    struct Case { size_t nx, ny, nz, pin, pout, maxVol; bool slice; };
    std::vector<Case> cases = {
        // single-cube (image fits in one padded cube)
        {48, 24, 16, 8, 0, 100000, false},
        {33, 15, 9, 4, 0, 10000,  false},
        {12, 15, 9, 2, 0, 5000,   true},
        {8, 15, 16,  2, 0, 3000,  false},
        {20, 24, 16, 8, 8, 5000000, true},
        // multi-cube (maxVolume forces several cubes)
        {20, 24, 16, 8, 8, 20000, true},
        {48, 24, 16, 8, 8, 30000, true},
        {20, 24, 16, 8, 0, 20000, false},
        {32, 32, 32, 8, 8, 25000, true},
        {24, 24, 24, 4, 2, 12000, false},
        {16, 40, 16, 4, 0, 10000, true},
        {40, 16, 16, 4, 4, 10000, false},
    };
    size_t checked = 0, multiCube = 0;
    for (const auto& c : cases) {
        CuboidShape image(c.nx, c.ny, c.nz);
        Padding inside{CuboidShape(c.pin, c.pin, c.pin), CuboidShape(c.pin, c.pin, c.pin)};
        Padding outside{CuboidShape(c.pout, c.pout, c.pout), CuboidShape(c.pout, c.pout, c.pout)};
        CuboidShape minShape = inside.getTotalPadding() + 1;
        if (minShape.getVolume() > c.maxVol) continue;
        auto r = splitImageHomogeneous(inside, outside, image, c.maxVol, 1, minShape, c.slice);
        if (!r.success) continue;
        ++checked;
        if (r.value.size() > 1) ++multiCube;
        CuboidShape T = r.value[0].getPaddedShape();
        expectContract({image, T, inside, outside, r.value});
    }
    GTEST_LOG_(INFO) << "Homogeneous_AnyShape: " << checked << " splits verified, "
        << multiCube << " multi-cube";
    EXPECT_GE(checked, 8u);
    EXPECT_GE(multiCube, 3u) << "sweep must exercise multi-cube homogeneous splits";
}

// Proves the exact-once tiling detector is not vacuous: a deliberately broken
// split (a 2px gap between two cores) must register as "not exactly once".
TEST_F(ImageSplitContractTest, CheckerDetectsGap) {
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
