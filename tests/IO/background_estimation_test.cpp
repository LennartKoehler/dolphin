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
// identical no matter how the image is chunked, and must recover the known
// background level of the image.

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

TEST_F(BackgroundEstimationTest, MatchesBackgroundLevelOfTheImage) {
    const Image3D image = makeImage();
    const std::string path = writeTestImage(dir + "/image.tif", image);

    TiffReader reader(path);
    reader.configure(0, ReaderConfig{});
    reader.setMemoryLimit(SIZE_MAX);

    const CuboidShape imageShape = reader.getMetaData().getShape();
    const float estimated = estimateImageBackground(
        reader, imageShape, CuboidShape{32, 32, 8}, /*threads=*/2);

    // The mode of the dark cluster recovers the known background level (0.05);
    // ~98% of the voxels sit there, so the cluster is unambiguous.
    const float expected = 0.05f;
    EXPECT_NEAR(estimated, expected, 0.05f * expected)
        << "streamed background " << estimated << " vs true level " << expected;
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
        reader, imageShape, CuboidShape{48, 40, 24}, 1);
    const float flat = estimateImageBackground(
        reader, imageShape, CuboidShape{48, 40, 1}, 4);

    EXPECT_FLOAT_EQ(tall, flat) << "chunking must not change the whole-image estimate";
}