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

#include <algorithm>
#include <cmath>
#include <random>

#include "dolphin_image/BackgroundStatistics.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"

namespace {

// Runs both streaming passes over `image`, exactly as the background
// estimators do over their read data.
float streamBackground(const Image3D& image) {
    return computeBackgroundStatistics([&](auto&& fn) { fn(image); }).background();
}

// Flat background with 2% camera noise, plus ~2% bright object voxels.
Image3D backgroundWithObjects(size_t w, size_t h, size_t d, float bg, unsigned seed) {
    Image3D image(CuboidShape(w, h, d), 0.0f);
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0f, bg * 0.02f);
    std::uniform_int_distribution<int> pick(0, 49);
    for (size_t z = 0; z < d; ++z)
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x) {
                const float base = bg + noise(gen);
                image.setPixel(x, y, z, pick(gen) == 0 ? base + 10.0f * bg : base);
            }
    return image;
}

} // namespace

TEST(BackgroundStatisticsTest, EmptyReturnsZero) {
    BackgroundStatistics stats;
    EXPECT_FLOAT_EQ(stats.background(), 0.0f);
    stats.setRange(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(stats.background(), 0.0f);
}

TEST(BackgroundStatisticsTest, ConstantImageReturnsConstant) {
    const float bg = 0.37f;
    Image3D image = Image3D(CuboidShape(8, 8, 8), bg);
    EXPECT_FLOAT_EQ(streamBackground(image), bg);
}

TEST(BackgroundStatisticsTest, MatchesBackgroundLevel) {
    const float bg = 0.05f;
    Image3D image = backgroundWithObjects(32, 24, 16, bg, 1234);
    const float actual = streamBackground(image);

    // The mode of the dark cluster recovers the noise floor's mean (mu) much
    // better than any low percentile (mu - 1.28 sigma or darker); remaining
    // error is bin resolution (~1% relative) and sampling.
    EXPECT_NEAR(actual, bg, 0.05f * bg)
        << "streaming background " << actual << " vs true level " << bg;
}

TEST(BackgroundStatisticsTest, ExplicitChunksMatchWholeImage) {
    // The same image split into z-chunks (as the plan streams it) must give the
    // same background as one shot: this is what prevents tile seams.
    const float bg = 0.05f;
    Image3D image = backgroundWithObjects(32, 24, 16, bg, 77);
    const float whole = streamBackground(image);

    BackgroundStatistics chunked;
    for (size_t z = 0; z < 16; z += 4) {
        Image3D chunk = image.getSubimageCopy(BoxCoord{CuboidPosition{0, 0, static_cast<int64_t>(z)},
                                                      CuboidShape{32, 24, 4}});
        forEachCountedVoxel(chunk,
                            [&chunked](float v) { chunked.addForRange(v); });
    }
    chunked.setRange(chunked.min(), chunked.max());
    for (size_t z = 0; z < 16; z += 4) {
        Image3D chunk = image.getSubimageCopy(BoxCoord{CuboidPosition{0, 0, static_cast<int64_t>(z)},
                                                      CuboidShape{32, 24, 4}});
        forEachCountedVoxel(chunk,
                            [&chunked](float v) { chunked.addForHistogram(v); });
    }
    EXPECT_FLOAT_EQ(chunked.background(), whole);
}

TEST(BackgroundStatisticsTest, IgnoresNonFiniteSamples) {
    BackgroundStatistics stats;
    stats.addForRange(1.0f);
    stats.addForRange(std::nanf(""));
    stats.addForRange(INFINITY);
    stats.addForRange(-INFINITY);
    stats.addForRange(2.0f);
    EXPECT_EQ(stats.count(), 2u);
    stats.setRange(stats.min(), stats.max());
    stats.addForHistogram(1.0f);
    stats.addForHistogram(2.0f);
    stats.addForHistogram(std::nanf(""));
    EXPECT_GE(stats.background(), 1.0f);
    EXPECT_LE(stats.background(), 2.0f);
}
// Regression: a tiny fraction of exact-zero voxels (dead pixels) must not
// collapse the estimate to 0. With min = 0 the old histogram anchored on
// kMinQuantile = 1e-12, so the bins spanned [1e-12, 1e-11], every real value
// clamped into the last bin (representative ~1e-11) and background() returned
// ~0.0000 - even though 99.99% of the image sat at the real dark level.
TEST(BackgroundStatisticsTest, FewExactZerosDoNotCollapseEstimate) {
    const float bg = 0.05f;
    // 200 x 200 x 20 = 800k voxels; 260 exact zeros (~0.03%), the user's case.
    Image3D image = backgroundWithObjects(200, 200, 20, bg, 99);
    const size_t zeros = 260;
    for (size_t i = 0; i < zeros; ++i) {
        image.setPixel(i % 200, (i / 200) % 200, i / 40000, 0.0f);
    }

    const float actual = streamBackground(image);
    EXPECT_GT(actual, 0.9f * bg)
        << "estimate collapsed to " << actual << " despite the background at " << bg;

    // The mode of the dark cluster stays at the true level; the handful of
    // zeros lives in its own bin and does not shift it.
    EXPECT_NEAR(actual, bg, 0.05f * bg)
        << "streaming background " << actual << " vs true level " << bg;
}

// The user's real-data case: the background occupies only a small minority of
// the image (here 3%), the rest is brighter haze. A percentile at ANY level
// below 97% lands in the foreground (measured: ~120 when the background is 80,
// even at the 5th percentile). The background is still the darkest dense
// cluster, so the estimate must find that cluster's mode instead.
TEST(BackgroundStatisticsTest, MinorityBackgroundStillFoundByMode) {
    const float bg = 80.0f;
    // 400 x 400 x 20 = 3.2M voxels; 3% stay at the background (+11% noise),
    // the remaining 97% are bright haze spread over [150, 600].
    Image3D image(CuboidShape(400, 400, 20), 0.0f);
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, bg * 0.11f);
    std::uniform_real_distribution<float> haze(150.0f, 600.0f);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    for (size_t z = 0; z < 20; ++z)
        for (size_t y = 0; y < 400; ++y)
            for (size_t x = 0; x < 400; ++x) {
                image.setPixel(x, y, z,
                               coin(gen) < 0.03f ? bg + noise(gen) : haze(gen));
            }

    const float actual = streamBackground(image);
    EXPECT_NEAR(actual, bg, 0.1f * bg)
        << "background is a 3% minority at " << bg << ", estimate was " << actual;
}

// Regression: a small cluster of POSITIVE dark values (not zeros) below the
// background must not collapse the estimate. With the old 1-decade range
// anchored on the smallest positive value, the whole bin window [lo, 10*lo]
// sat BELOW the background, so every real background voxel clamped into the
// last bin (representative ~10*lo) and background() returned far too low.
// Percentile-over-full-range must be immune to this.
TEST(BackgroundStatisticsTest, LowPositiveOutliersDoNotCollapseEstimate) {
    const float bg = 0.05f;
    // 100 x 100 x 20 = 200k voxels at the background, plus 400 dark-positive
    // outliers ~2 decades below (a dark region / low-value artifact).
    Image3D image = backgroundWithObjects(100, 100, 20, bg, 7);
    const size_t volume = 100 * 100 * 20;
    const size_t outliers = std::min<size_t>(400, volume / 2);
    for (size_t i = 0; i < outliers; ++i) {
        image.setPixel(i % 100, (i / 100) % 100, i / 10000, bg * 0.005f);
    }

    const float actual = streamBackground(image);
    EXPECT_GT(actual, 0.9f * bg)
        << "estimate collapsed to " << actual << " despite the background at " << bg;
    EXPECT_LT(actual, 1.1f * bg)
        << "estimate inflated to " << actual << " against the background at " << bg;
}

// The other side of the rule: when zeros genuinely dominate the dark fraction
// (masked / empty regions), the background IS zero and must stay 0.
TEST(BackgroundStatisticsTest, MajorityZerosKeepZeroBackground) {
    const float bg = 0.05f;
    Image3D image = backgroundWithObjects(50, 50, 20, bg, 5);
    // Zero out 60% of the voxels: far more than the 10% dark fraction.
    const size_t volume = 50 * 50 * 20;
    for (size_t i = 0; i < volume * 6 / 10; ++i) {
        image.setPixel(i % 50, (i / 50) % 50, i / 2500, 0.0f);
    }

    const float actual = streamBackground(image);
    EXPECT_LT(actual, 1e-6f) << "majority-zero image estimated background " << actual;
}
