#include <gtest/gtest.h>
#include "dolphin/Dolphin.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/Logging.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "TestUtils.h"
#include <filesystem>

class MainIntegrationTest : public ::testing::Test {
protected:
    std::string testDir;

    void SetUp() override {
        testDir = TestUtils::outputPath();
        Logging::init();
    }
};

TEST_F(MainIntegrationTest, DolphinInit) {
    Dolphin dolphin;
    EXPECT_NO_THROW(dolphin.init(testDir));
}

TEST_F(MainIntegrationTest, DolphinGeneratePSF) {
    Dolphin dolphin;
    dolphin.init(testDir);

    SetupConfigPSF setupConfig;
    setupConfig.backend = "cpu";
    setupConfig.nThreads = 1;
    setupConfig.nIOThreads = 1;
    setupConfig.nWorkerThreads = 1;
    setupConfig.nDevices = 1;
    setupConfig.maxMem_GB = 1;
    setupConfig.psfConfigPath = TestUtils::outputPath() + "/gaussian_psf.json";

    std::ofstream file(setupConfig.psfConfigPath);
    file << TestUtils::gaussianPSFConfigJSON();
    file.close();

    setupConfig.outputPath = testDir + "/generated_psf.tif";

    PSFGenerationRequest request;
    request.setConfig(std::make_shared<SetupConfigPSF>(setupConfig));

    auto result = dolphin.generatePSF(request);
    ASSERT_NE(result, nullptr);
}

TEST_F(MainIntegrationTest, SetupConfigLoadAndValidate) {
    auto configPath = testDir + "/integration_config.json";
    std::ofstream file(configPath);
    file << TestUtils::defaultSetupConfigJSON();
    file.close();

    auto setupConfig = SetupConfig::createFromJSONFile(configPath);
    EXPECT_EQ(setupConfig.backend, "cpu");
    EXPECT_EQ(setupConfig.nIOThreads, 1);
}

TEST_F(MainIntegrationTest, DeconvolutionConfigLoad) {
    auto configPath = testDir + "/deconv_config.json";
    std::ofstream file(configPath);
    file << TestUtils::defaultDeconvConfigJSON();
    file.close();

    auto config = DeconvolutionConfig::createFromJSONFile(configPath);
    EXPECT_EQ(config.algorithmName, "RichardsonLucy");
    EXPECT_EQ(config.iterations, 10);
}
