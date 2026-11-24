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

// End-to-end: PoissonMLE on an image split into many cubes. The plan resolves
// the background once for the whole image (background_strategy = global) and
// every cube is deconvolved against that value. This exercises the whole path:
// createPlan -> streaming background estimate -> DeconvolutionProcessor ->
// deconvolve(runContext) -> writer.

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>

#include "dolphin/Dolphin.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"
#include "dolphin/Logging.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "TestUtils.h"

namespace {

// Background + noise with a few objects, so the low percentile is well defined.
Image3D makeInput(size_t w, size_t h, size_t d) {
    Image3D image = TestUtils::createRandomImage(w, h, d);
    for (auto& v : image) v = std::abs(v) * 0.5f + 0.02f;
    image.setPixel(w / 4, h / 4, d / 4, 1.0f);
    image.setPixel(w / 2, h / 2, d / 2, 1.0f);
    return image;
}

void runPoissonMleE2E(BackgroundStrategyType strategy) {
    const std::string testDir = TestUtils::outputPath() + "/e2e_poisson_bg";
    std::filesystem::create_directories(testDir);

    Dolphin dolphin;
    dolphin.init(testDir);

    const size_t w = 64, h = 64, d = 32;
    const std::string imagePath = testDir + "/input.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(imagePath, makeInput(w, h, d)));

    // Small cubes force several tasks; all must share one background.
    GaussianPSFConfig psfConfig;
    psfConfig.psfModelName = "Gaussian";
    psfConfig.ID = "e2e_poisson_psf";
    psfConfig.sizeX = 15;
    psfConfig.sizeY = 15;
    psfConfig.sizeZ = 15;
    psfConfig.sigmaX = 2;
    psfConfig.sigmaY = 2;
    psfConfig.sigmaZ = 2;

    SetupConfig setupConfig;
    setupConfig.imagePath = imagePath;
    setupConfig.outputPath = testDir + "/result.tif";
    setupConfig.backend = "cpu";
    setupConfig.threadConfig.nIOThreads = 2;
    setupConfig.threadConfig.nWorkerThreads = 2;
    setupConfig.nDevices = 1;
    setupConfig.maxMemHost_gb = 8;
    setupConfig.maxMemDevice_gb = 8;
    setupConfig.psfConfigs = {std::make_shared<GaussianPSFConfig>(psfConfig)};

    auto deconvConfig = DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm("PoissonMLE");
    auto& poisson = static_cast<PoissonMLEConfig&>(*deconvConfig);
    poisson.iterations = 3;
    poisson.signalToNoise = 10.0f;
    poisson.minDenominator = 1e-6f;
    poisson.paddingFillType = PaddingFillType::LINEAR;
    poisson.paddingStrategyType = PaddingStrategyType::LOW;
    poisson.backgroundStrategyType = strategy;
    poisson.cubeSize = {32, 32, 32};

    std::atomic<float> progress{0.0f};
    DeconvolutionRequest request(
        std::make_shared<SetupConfig>(setupConfig), deconvConfig,
        [](std::atomic<float>& counter, float max) { counter.store(max); });

    const auto result = dolphin.deconvolve(request);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->success()) << "PoissonMLE deconvolution failed: " << result->errorMessage();

    const auto output = TiffReader::readTiffFile(setupConfig.outputPath, 0);
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->getShape(), CuboidShape(w, h, d));
    EXPECT_FALSE(TestUtils::hasNaN(*output));
    EXPECT_FALSE(TestUtils::hasInf(*output));
}

} // namespace

TEST(PoissonMLEEndToEndTest, GlobalBackgroundAcrossCubes) {
    TestUtils::initLogging();
    runPoissonMleE2E(BACKGROUND_GLOBAL);
}

TEST(PoissonMLEEndToEndTest, WholeImageBackgroundAcrossCubes) {
    TestUtils::initLogging();
    runPoissonMleE2E(BACKGROUND_WHOLE_IMAGE);
}