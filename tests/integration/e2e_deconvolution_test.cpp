#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <atomic>

#include "dolphin/Dolphin.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/Logging.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/IO/TiffReader.h"
#include "TestUtils.h"

class EndToEndTest : public ::testing::Test {
protected:
    std::string testDir;
    std::unique_ptr<Dolphin> dolphin;

    void SetUp() override {
        testDir = "/tmp/dolphin/e2e_test";
        std::filesystem::create_directories(testDir);
        dolphin = std::make_unique<Dolphin>();
        dolphin->init(testDir);
    }

    static void progressCallback(std::atomic<float>& current, float max) {
        float progress = (current * 100.0f) / max;
        GTEST_LOG_(INFO) << "Progress: " << progress << "%";
    }
};

TEST_F(EndToEndTest, FullDeconvolutionPipeline) {
    GTEST_LOG_(INFO) << "=== End-to-End Deconvolution Test ===";

    // --- Step 1: PSF config round-trip ---
    GTEST_LOG_(INFO) << "Step 1: PSF config write/read round-trip";

    GaussianPSFConfig psfConfig;
    psfConfig.psfModelName = "Gaussian";
    psfConfig.ID = "e2e_gaussian";
    psfConfig.sizeX = 32;
    psfConfig.sizeY = 32;
    psfConfig.sizeZ = 16;
    psfConfig.sigmaX = 5;
    psfConfig.sigmaY = 5;
    psfConfig.sigmaZ = 5;
    psfConfig.resLateral_nm = 5000;
    psfConfig.resAxial_nm = 5000;

    std::string psfConfigPath = testDir + "/psf_config.json";
    {
        ordered_json j = psfConfig.writeToJSON();
        std::ofstream o(psfConfigPath);
        o << std::setw(4) << j << std::endl;
        ASSERT_TRUE(o.good()) << "Failed to write PSF config file";
    }

    std::shared_ptr<PSFConfig> readPsfConfig;
    {
        json jsonData;
        std::ifstream file(psfConfigPath);
        file >> jsonData;
        readPsfConfig = PSFConfig::createFromJSON(jsonData);
        ASSERT_NE(readPsfConfig, nullptr);
        EXPECT_EQ(readPsfConfig->getModelName(), "Gaussian");
        EXPECT_EQ(readPsfConfig->sizeX, 32u);
        EXPECT_EQ(readPsfConfig->sizeY, 32u);
        EXPECT_EQ(readPsfConfig->sizeZ, 16u);

        auto gaussianCfg = std::dynamic_pointer_cast<GaussianPSFConfig>(readPsfConfig);
        ASSERT_NE(gaussianCfg, nullptr);
        EXPECT_FLOAT_EQ(gaussianCfg->sigmaX, 5.0f);
        EXPECT_FLOAT_EQ(gaussianCfg->sigmaY, 5.0f);
        EXPECT_FLOAT_EQ(gaussianCfg->sigmaZ, 5.0f);
    }

    // --- Step 2: Image round-trip ---
    GTEST_LOG_(INFO) << "Step 2: Image write/read round-trip";

    const size_t imgW = 32, imgH = 32, imgD = 16;
    Image3D gradientImage = TestUtils::createGradientImage(imgW, imgH, imgD);

    std::string imagePath = testDir + "/gradient_input.tif";
    ASSERT_TRUE(TiffWriter::writeToFile(imagePath, gradientImage))
        << "Failed to write gradient image to TIFF";

    auto readImageOpt = TiffReader::readTiffFile(imagePath, 0);
    ASSERT_TRUE(readImageOpt.has_value()) << "Failed to read gradient image from TIFF";
    EXPECT_TRUE(readImageOpt->isEqual(gradientImage, 0.001f))
        << "Round-trip image does not match original";

    // --- Step 3: Setup + Deconvolution config round-trip ---
    GTEST_LOG_(INFO) << "Step 3: Setup/Deconv config write/read round-trip";

    std::string outputPath = testDir + "/deconv_result.tif";

    SetupConfig setupConfig;
    setupConfig.imagePath = imagePath;
    setupConfig.outputPath = outputPath;
    setupConfig.backend = "cpu";
    setupConfig.nThreads = 1;
    setupConfig.nIOThreads = 1;
    setupConfig.nWorkerThreads = 1;
    setupConfig.nDevices = 1;
    setupConfig.maxMemHost_gb = 8;
    setupConfig.maxMemDevice_gb = 8;
    setupConfig.multiplePsfConfigPaths = {psfConfigPath};
    setupConfig.savePsf = true;

    DeconvolutionConfig deconvConfig;
    deconvConfig.algorithmName = "RichardsonLucy";
    deconvConfig.iterations = 5;
    deconvConfig.epsilon = 1e-6f;
    deconvConfig.lambda = 0.001f;
    deconvConfig.paddingFillType = PaddingFillType::MIRROR;
    deconvConfig.paddingStrategyType = PaddingStrategyType::PARENT;

    std::string setupConfigPath = testDir + "/setup_config.json";
    {
        ordered_json setupJson = setupConfig.writeToJSON();
        ordered_json deconvJson = deconvConfig.writeToJSON();
        setupJson["deconvolution_config"] = deconvJson;

        std::ofstream o(setupConfigPath);
        o << std::setw(4) << setupJson << std::endl;
        ASSERT_TRUE(o.good()) << "Failed to write setup config file";
    }

    SetupConfig readSetupConfig;
    DeconvolutionConfig readDeconvConfig;
    {
        readSetupConfig = SetupConfig::createFromJSONFile(setupConfigPath);
        readDeconvConfig = DeconvolutionConfig::createFromJSONFile(setupConfigPath);

        EXPECT_EQ(readSetupConfig.imagePath, imagePath);
        EXPECT_EQ(readSetupConfig.outputPath, outputPath);
        EXPECT_EQ(readSetupConfig.backend, "cpu");
        ASSERT_EQ(readSetupConfig.multiplePsfConfigPaths.size(), 1u);
        EXPECT_EQ(readSetupConfig.multiplePsfConfigPaths[0], psfConfigPath);
        EXPECT_TRUE(readSetupConfig.savePsf);
        EXPECT_FLOAT_EQ(readSetupConfig.maxMemHost_gb, 8.0f);
        EXPECT_FLOAT_EQ(readSetupConfig.maxMemDevice_gb, 8.0f);

        EXPECT_EQ(readDeconvConfig.algorithmName, "RichardsonLucy");
        EXPECT_EQ(readDeconvConfig.iterations, 5);
        EXPECT_EQ(readDeconvConfig.paddingFillType, PaddingFillType::MIRROR);
        EXPECT_EQ(readDeconvConfig.paddingStrategyType, PaddingStrategyType::PARENT);
    }

    // --- Step 4: Deconvolution ---
    GTEST_LOG_(INFO) << "Step 4: Running deconvolution";

    DeconvolutionRequest request(
        std::make_shared<SetupConfig>(readSetupConfig),
        std::make_shared<DeconvolutionConfig>(readDeconvConfig),
        progressCallback);

    auto result = dolphin->deconvolve(request);

    // --- Step 5: Verify ---
    GTEST_LOG_(INFO) << "Step 5: Verifying results";

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->success())
        << "Deconvolution failed: " << result->errorMessage();

    EXPECT_TRUE(std::filesystem::exists(outputPath))
        << "Output TIFF file not found at: " << outputPath;

    auto readResultOpt = TiffReader::readTiffFile(outputPath, 0);
    ASSERT_TRUE(readResultOpt.has_value())
        << "Failed to read deconvolution result TIFF";
    EXPECT_EQ(readResultOpt->getShape(), CuboidShape(imgW, imgH, imgD))
        << "Result image dimensions mismatch";
    EXPECT_FALSE(TestUtils::hasNaN(*readResultOpt))
        << "Result image contains NaN values";
    EXPECT_FALSE(TestUtils::hasInf(*readResultOpt))
        << "Result image contains Inf values";

    GTEST_LOG_(INFO) << "=== End-to-End Test Completed Successfully ===";
}
