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
#include <numeric>
#include <random>
#include <vector>

#include "dolphin_image/BackgroundStatistics.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImagePadding.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"

namespace {

// Exact reference: mean of the dimmest kDarkFraction of `values`.
float referenceBackground(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t n = std::max<size_t>(1, static_cast<size_t>(
        std::ceil(static_cast<double>(values.size()) * BackgroundStatistics::kDarkFraction)));
    const double sum = std::accumulate(values.begin(), values.begin() + static_cast<long>(n), 0.0);
    return static_cast<float>(sum / static_cast<double>(n));
}

// Runs both streaming passes over `image`, exactly as estimateImageBackground
// does for one chunk.
float streamBackground(const Image3D& image, const Padding& padding, PaddingFillType fill) {
    BackgroundStatistics stats;
    forEachCountedVoxel(image, padding, fill, BackgroundStatisticsDetail::RangeAccumulator{stats});
    stats.setRange(stats.min(), stats.max());
    forEachCountedVoxel(image, padding, fill, BackgroundStatisticsDetail::HistogramAccumulator{stats});
    return stats.background();
}

std::vector<float> valuesOf(const Image3D& image) {
    std::vector<float> values;
    values.reserve(image.getShape().getVolume());
    for (const auto& v : image) values.push_back(static_cast<float>(v));
    return values;
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

const Padding kNoPadding{CuboidShape{0, 0, 0}, CuboidShape{0, 0, 0}};

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
    EXPECT_FLOAT_EQ(streamBackground(image, kNoPadding, PaddingFillType::ZERO), bg);
}

TEST(BackgroundStatisticsTest, MatchesExactDimmestDecile) {
    const float bg = 0.05f;
    Image3D image = backgroundWithObjects(32, 24, 16, bg, 1234);
    const float expected = referenceBackground(valuesOf(image));
    const float actual = streamBackground(image, kNoPadding, PaddingFillType::ZERO);

    // Log bins (~1% relative resolution) on a narrow, unimodal background.
    EXPECT_NEAR(actual, expected, 0.02f * bg)
        << "streaming background " << actual << " vs exact " << expected;
}

TEST(BackgroundStatisticsTest, ExplicitChunksMatchWholeImage) {
    // The same image split into z-chunks (as the plan streams it) must give the
    // same background as one shot: this is what prevents tile seams.
    const float bg = 0.05f;
    Image3D image = backgroundWithObjects(32, 24, 16, bg, 77);
    const float whole = streamBackground(image, kNoPadding, PaddingFillType::ZERO);

    BackgroundStatistics chunked;
    for (size_t z = 0; z < 16; z += 4) {
        Image3D chunk = image.getSubimageCopy(BoxCoord{CuboidPosition{0, 0, static_cast<int64_t>(z)},
                                                      CuboidShape{32, 24, 4}});
        forEachCountedVoxel(chunk, kNoPadding, PaddingFillType::ZERO,
                            BackgroundStatisticsDetail::RangeAccumulator{chunked});
    }
    chunked.setRange(chunked.min(), chunked.max());
    for (size_t z = 0; z < 16; z += 4) {
        Image3D chunk = image.getSubimageCopy(BoxCoord{CuboidPosition{0, 0, static_cast<int64_t>(z)},
                                                      CuboidShape{32, 24, 4}});
        forEachCountedVoxel(chunk, kNoPadding, PaddingFillType::ZERO,
                            BackgroundStatisticsDetail::HistogramAccumulator{chunked});
    }
    EXPECT_FLOAT_EQ(chunked.background(), whole);
}

TEST(BackgroundStatisticsTest, GradientPaddingDoesNotBiasEstimate) {
    // A linear (fading-to-zero) fill pads with values below the background. Those
    // padded voxels must not be counted, otherwise every image-border chunk would
    // pull the estimate down.
    const float bg = 0.2f;
    Image3D interior = backgroundWithObjects(16, 16, 16, bg, 5);
    const float expected = referenceBackground(valuesOf(interior));
    EXPECT_GT(expected, 0.9f * bg);

    Padding padding;
    padding.before = CuboidShape(4, 4, 4);
    padding.after = CuboidShape(4, 4, 4);
    Image3D padded = interior;
    ImagePadding::padImage(padded, padding, PaddingFillType::LINEAR);

    // Counting the padded chunk with the linear fill must skip the border.
    const float actual = streamBackground(padded, padding, PaddingFillType::LINEAR);
    EXPECT_NEAR(actual, expected, 0.02f * bg);

    // Sanity: with a ZERO fill the (true) padding zeros ARE counted.
    Image3D zeroPadded = interior;
    ImagePadding::padImage(zeroPadded, padding, PaddingFillType::ZERO);
    const float withZeros = streamBackground(zeroPadded, padding, PaddingFillType::ZERO);
    EXPECT_LT(withZeros, 0.5f * bg) << "zero padding should count as background";
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
TEST(BackgroundStrategyConfigTest, LoadsFromJSONAndCopies) {
    auto config = DeconvolutionConfig::createFromJSON(nlohmann::json::parse(R"({
        "algorithm_name": "PoissonMLE",
        "background_strategy": "whole_image"
    })"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->backgroundStrategyType, BACKGROUND_WHOLE_IMAGE);

    // Copy semantics keep the parameter registered and its value.
    const auto copy = std::make_shared<PoissonMLEConfig>(
        static_cast<const PoissonMLEConfig&>(*config));
    EXPECT_EQ(copy->backgroundStrategyType, BACKGROUND_WHOLE_IMAGE);
    const auto written = copy->writeToJSON();
    ASSERT_TRUE(written.contains("background_strategy"));
    EXPECT_EQ(written["background_strategy"].get<std::string>(), "whole_image");
}

TEST(BackgroundStrategyConfigTest, DefaultsToGlobal) {
    auto config = DeconvolutionConfig::createFromJSON(nlohmann::json::parse(R"({
        "algorithm_name": "PoissonMLE"
    })"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->backgroundStrategyType, BACKGROUND_GLOBAL);
}
