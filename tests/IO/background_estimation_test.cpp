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

// The whole-image background is streamed from the real reader during plan
// creation. This test runs that path end to end on a TIFF: the estimate must be
// identical no matter how the image is chunked, and must match the exact
// dimmest decile of the image.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>
#include <vector>

#include "dolphin/deconvolution/BackgroundEstimation.h"
#include "dolphin/Logging.h"
#include "dolphin_image/BackgroundStatistics.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "TestUtils.h"

namespace {

std::string writeTestImage(const std::string& path, const Image3D& image) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    EXPECT_TRUE(TiffWriter::writeToFile(path, image, WriterCompressionConfig{}, WriterConfig{}));
    return path;
}

float referenceDimmestDecile(const Image3D& image) {
    std::vector<float> values;
    values.reserve(image.getShape().getVolume());
    for (const auto& v : image) values.push_back(static_cast<float>(v));
    std::sort(values.begin(), values.end());
    const size_t n = std::max<size_t>(1, static_cast<size_t>(
        std::ceil(static_cast<double>(values.size()) * BackgroundStatistics::kDarkFraction)));
    return static_cast<float>(std::accumulate(values.begin(), values.begin() + static_cast<long>(n), 0.0) / n);
}

} // namespace

class BackgroundEstimationTest : public ::testing::Test {
protected:
    void SetUp() override {
        TestUtils::initLogging();
        dir = TestUtils::outputPath() + "/bg_estimation";
        std::filesystem::create_directories(dir);
    }

    Image3D makeImage() const {
        // 3% background with 2% camera noise plus sparse bright objects.
        const size_t w = 48, h = 40, d = 24;
        const float bg = 0.05f;
        Image3D image(CuboidShape(w, h, d), 0.0f);
        std::mt19937 gen(7);
        std::normal_distribution<float> noise(0.0f, bg * 0.02f);
        std::uniform_int_distribution<int> pick(0, 49);
        for (size_t z = 0; z < d; ++z)
            for (size_t y = 0; y < h; ++y)
                for (size_t x = 0; x < w; ++x) {
                    float v = bg + noise(gen);
                    if (pick(gen) == 0) v += 20.0f * bg;
                    image.setPixel(x, y, z, v);
                }
        return image;
    }

    std::string dir;
};

TEST_F(BackgroundEstimationTest, MatchesExactDimmestDecileOfTheImage) {
    const Image3D image = makeImage();
    const std::string path = writeTestImage(dir + "/image.tif", image);

    TiffReader reader(path);
    reader.configure(0, ReaderConfig{});
    reader.setMemoryLimit(SIZE_MAX);

    const CuboidShape imageShape = reader.getMetaData().getShape();
    // The padding fill is an interior fill; borders are dropped while streaming.
    const float estimated = estimateImageBackground(
        reader, imageShape, CuboidShape{32, 32, 8}, PaddingFillType::LINEAR, /*threads=*/2);

    const float expected = referenceDimmestDecile(image);
    EXPECT_NEAR(estimated, expected, 0.02f * 0.05f)
        << "streamed background " << estimated << " vs exact " << expected;
}

TEST_F(BackgroundEstimationTest, IndependentOfChunkSize) {
    const Image3D image = makeImage();
    const std::string path = writeTestImage(dir + "/chunked.tif", image);

    TiffReader reader(path);
    reader.configure(0, ReaderConfig{});
    reader.setMemoryLimit(SIZE_MAX);
    const CuboidShape imageShape = reader.getMetaData().getShape();

    // The chunk height is derived from the cube hint: a large hint gives few,
    // tall chunks, a small one many flat chunks. Both must agree exactly.
    const float tall = estimateImageBackground(
        reader, imageShape, CuboidShape{48, 40, 24}, PaddingFillType::MIRROR, 1);
    const float flat = estimateImageBackground(
        reader, imageShape, CuboidShape{48, 40, 1}, PaddingFillType::MIRROR, 4);

    EXPECT_FLOAT_EQ(tall, flat) << "chunking must not change the whole-image estimate";
}