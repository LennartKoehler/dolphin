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
#include "dolphin/DefaultParameters.h"
#include "dolphin/Logging.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/ThreadPool.h"
#include "TestUtils.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using json = nlohmann::json;

// The DefaultParameters singleton is created lazily on the first
// getDefaultParameters() call, so the env var must be set in SetUpTestSuite
// (before any test body runs) and left set. This binary is separate from the
// unit tests because the singleton persists for the whole process.
class DefaultParametersOverrideTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        Logging::init();
        auto path = std::filesystem::path(TestUtils::outputPath()) / "default_parameters_override.json";
        json override = {
            {"low_padding_threshold", 0.5},
            {"medium_padding_threshold", 0.25},
            {"generation_threshold_multiplier", 0.125}
        };
        {
            std::ofstream file(path);
            file << override.dump();
        }
        setenv("DOLPHIN_DEFAULT_PARAMETERS", path.string().c_str(), 1);
    }
};

TEST_F(DefaultParametersOverrideTest, EnvVarOverrideWins) {
    const auto& params = DefaultParameters::getDefaultParameters();
    EXPECT_FLOAT_EQ(params.lowPaddingThreshold(), 0.5f);
    EXPECT_FLOAT_EQ(params.mediumPaddingThreshold(), 0.25f);
    EXPECT_FLOAT_EQ(params.generationThresholdMultiplier(), 0.125f);
}

// Proves PSFHandler reads the thresholds from DefaultParameters rather than
// hardcoded constants. With MEDIUM the generation cutoff is
// 0.25 * 0.125 = 0.03125, giving a Gaussian support of
// 2*ceil(sqrt(-2*ln(0.03125))*5)+1 = 29 px. The old hardcoded MEDIUM cutoff
// (1e-6) would instead give 55 px, so 29 discriminates the two.
TEST_F(DefaultParametersOverrideTest, PresetPaddingUsesDefaults) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    j["size_x"] = 0;
    j["size_y"] = 0;
    j["size_z"] = 0;
    auto psfConfig = factory.createConfig(j);

    auto threadPool = std::make_shared<ThreadPool>(1);
    PSFHandler handler(threadPool, [](std::atomic<float>&, float){});

    SetupConfig setupConfig;
    setupConfig.psfConfigs = {psfConfig};
    setupConfig.psfFilePaths = {};

    DeconvolutionConfig deconvConfig;
    deconvConfig.paddingStrategyType = PaddingStrategyType::MEDIUM;

    handler.generatePSFs(setupConfig, deconvConfig, CuboidShape{128, 128, 64});
    EXPECT_EQ(handler.getMaxShape().width, 29u);
}
