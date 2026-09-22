#include <gtest/gtest.h>
#include "dolphin/Config.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>

using json = nlohmann::json;

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(ConfigTest, DeconvolutionConfigDefaults) {
    DeconvolutionConfig config;
    EXPECT_EQ(config.algorithmName, "RichardsonLucy");
    EXPECT_EQ(config.iterations, 10);
    EXPECT_FLOAT_EQ(config.epsilon, 1e-6f);
    EXPECT_FLOAT_EQ(config.lambda, 0.001f);
    EXPECT_EQ(config.paddingFillType, PaddingFillType::ZERO);
    EXPECT_EQ(config.paddingStrategyType, PaddingStrategyType::PARENT);
}

TEST_F(ConfigTest, DeconvolutionConfigLoadFromJSON) {
    auto jsonStr = TestUtils::defaultDeconvConfigJSON();
    json j = json::parse(jsonStr);
    DeconvolutionConfig config;
    ASSERT_TRUE(config.loadFromJSON(j));
    EXPECT_EQ(config.algorithmName, "RichardsonLucy");
    EXPECT_EQ(config.iterations, 10);
    EXPECT_FLOAT_EQ(config.epsilon, 1e-6f);
    EXPECT_FLOAT_EQ(config.lambda, 0.001f);
}

TEST_F(ConfigTest, DeconvolutionConfigWriteToJSON) {
    DeconvolutionConfig config;
    config.algorithmName = "InverseFilter";
    config.iterations = 50;
    json j = config.writeToJSON();
    ASSERT_TRUE(j.contains("algorithm_name"));
    EXPECT_EQ(j["algorithm_name"], "InverseFilter");
    EXPECT_EQ(j["iterations"], 50);
}

TEST_F(ConfigTest, DeconvolutionConfigRoundTrip) {
    DeconvolutionConfig config1;
    config1.algorithmName = "RichardsonLucyTotalVariation";
    config1.iterations = 25;
    config1.lambda = 0.05f;
    json j = config1.writeToJSON();

    DeconvolutionConfig config2;
    ASSERT_TRUE(config2.loadFromJSON(j));
    EXPECT_EQ(config2.algorithmName, "RichardsonLucyTotalVariation");
    EXPECT_EQ(config2.iterations, 25);
    EXPECT_FLOAT_EQ(config2.lambda, 0.05f);
}

TEST_F(ConfigTest, DeconvolutionConfigCopyConstructor) {
    DeconvolutionConfig config1;
    config1.iterations = 42;
    DeconvolutionConfig config2(config1);
    EXPECT_EQ(config2.iterations, 42);
}

TEST_F(ConfigTest, DeconvolutionConfigAssignment) {
    DeconvolutionConfig config1;
    config1.iterations = 99;
    DeconvolutionConfig config2;
    config2 = config1;
    EXPECT_EQ(config2.iterations, 99);
}

TEST_F(ConfigTest, DeconvolutionConfigCreateFromJSONFile) {
    auto jsonStr = TestUtils::defaultDeconvConfigJSON();
    auto filePath = TestUtils::outputPath() + "/deconv_config.json";
    std::ofstream file(filePath);
    file << jsonStr;
    file.close();

    auto config = DeconvolutionConfig::createFromJSONFile(filePath);
    EXPECT_EQ(config.algorithmName, "RichardsonLucy");
    EXPECT_EQ(config.iterations, 10);
}

TEST_F(ConfigTest, DeconvolutionConfigMissingFile) {
    EXPECT_THROW(DeconvolutionConfig::createFromJSONFile("/nonexistent/path.json"), std::exception);
}

TEST_F(ConfigTest, SetupConfigDefaults) {
    SetupConfig config;
    EXPECT_EQ(config.backend, "cpu");
    EXPECT_EQ(config.nThreads, 1);
    EXPECT_EQ(config.nIOThreads, 1);
    EXPECT_EQ(config.nWorkerThreads, 1);
    EXPECT_EQ(config.nDevices, 1);
    EXPECT_EQ(config.savePsf, false);
}

TEST_F(ConfigTest, SetupConfigLoadFromJSON) {
    auto jsonStr = TestUtils::defaultSetupConfigJSON();
    json j = json::parse(jsonStr);
    SetupConfig config;
    ASSERT_TRUE(config.loadFromJSON(j));
    EXPECT_EQ(config.backend, "cpu");
    EXPECT_EQ(config.nIOThreads, 1);
    EXPECT_EQ(config.nWorkerThreads, 1);
    EXPECT_EQ(config.imagePath, "test_input.tif");
}

TEST_F(ConfigTest, SetupConfigGetDeconvTypeStandard) {
    SetupConfig config;
    config.labeledImage = "";
    EXPECT_EQ(config.getDeconvType(), DeconvolutionType::STANDARD);
}

TEST_F(ConfigTest, SetupConfigGetDeconvTypeLabeled) {
    SetupConfig config;
    config.labeledImage = "label.tif";
    EXPECT_EQ(config.getDeconvType(), DeconvolutionType::LABELED);
}

TEST_F(ConfigTest, SetupConfigCreateFromJSONFile) {
    auto jsonStr = TestUtils::defaultSetupConfigJSON();
    auto filePath = TestUtils::outputPath() + "/setup_config.json";
    std::ofstream file(filePath);
    file << jsonStr;
    file.close();

    auto config = SetupConfig::createFromJSONFile(filePath);
    EXPECT_EQ(config.backend, "cpu");
    EXPECT_EQ(config.imagePath, "test_input.tif");
}

TEST_F(ConfigTest, SetupConfigMissingFile) {
    EXPECT_THROW(SetupConfig::createFromJSONFile("/nonexistent/path.json"), std::exception);
}

TEST_F(ConfigTest, SetupConfigCopyConstructor) {
    SetupConfig config1;
    config1.backend = "cuda";
    config1.nThreads = 8;
    SetupConfig config2(config1);
    EXPECT_EQ(config2.backend, "cuda");
    EXPECT_EQ(config2.nThreads, 8);
}

TEST_F(ConfigTest, SetupConfigAssignment) {
    SetupConfig config1;
    config1.nIOThreads = 5;
    SetupConfig config2;
    config2 = config1;
    EXPECT_EQ(config2.nIOThreads, 5);
}

TEST_F(ConfigTest, GaussianPSFConfigDefaults) {
    GaussianPSFConfig config;
    EXPECT_EQ(config.psfModelName, "Gaussian");
    EXPECT_EQ(config.sizeX, 20);
    EXPECT_EQ(config.sizeY, 20);
    EXPECT_EQ(config.sizeZ, 10);
    EXPECT_FLOAT_EQ(config.sigmaX, 10);
    EXPECT_FLOAT_EQ(config.sigmaY, 10);
    EXPECT_FLOAT_EQ(config.sigmaZ, 10);
}

TEST_F(ConfigTest, GaussianPSFConfigLoadFromJSON) {
    auto jsonStr = TestUtils::gaussianPSFConfigJSON();
    json j = json::parse(jsonStr);
    auto config = PSFConfig::createFromJSON(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModelName(), "Gaussian");
    EXPECT_EQ(config->sizeX, 32);
    EXPECT_EQ(config->sizeY, 32);
    EXPECT_EQ(config->sizeZ, 16);

    auto* gaussConfig = dynamic_cast<GaussianPSFConfig*>(config.get());
    ASSERT_NE(gaussConfig, nullptr);
    EXPECT_FLOAT_EQ(gaussConfig->sigmaX, 5);
}

TEST_F(ConfigTest, GibsonLanniPSFConfigLoadFromJSON) {
    auto jsonStr = TestUtils::gibsonLanniPSFConfigJSON();
    json j = json::parse(jsonStr);
    auto config = PSFConfig::createFromJSON(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModelName(), "GibsonLanni");
    EXPECT_EQ(config->sizeX, 64);
    EXPECT_EQ(config->sizeY, 64);
    EXPECT_EQ(config->sizeZ, 32);
    EXPECT_FLOAT_EQ(config->NA, 1.4f);

    auto* glConfig = dynamic_cast<GibsonLanniPSFConfig*>(config.get());
    ASSERT_NE(glConfig, nullptr);
    EXPECT_FLOAT_EQ(glConfig->lambda_nm, 450.0f);
}

TEST_F(ConfigTest, PSFConfigInvalidModel) {
    json j = {{"model_name", "NonExistent"}, {"size_x", 10}, {"size_y", 10}, {"size_z", 10}};
    EXPECT_THROW(PSFConfig::createFromJSON(j), std::runtime_error);
}

TEST_F(ConfigTest, PSFConfigMissingModelName) {
    json j = {{"size_x", 10}, {"size_y", 10}, {"size_z", 10}};
    EXPECT_THROW(PSFConfig::createFromJSON(j), std::runtime_error);
}

TEST_F(ConfigTest, PSFConfigCompareDim) {
    GaussianPSFConfig a;
    a.sizeX = 32; a.sizeY = 32; a.sizeZ = 16;

    GaussianPSFConfig b;
    b.sizeX = 32; b.sizeY = 32; b.sizeZ = 16;

    EXPECT_TRUE(a.compareDim(b));

    b.sizeZ = 20;
    EXPECT_FALSE(a.compareDim(b));
}

TEST_F(ConfigTest, GaussianPSFConfigCopyConstructor) {
    GaussianPSFConfig config1;
    config1.sigmaX = 7;
    GaussianPSFConfig config2(config1);
    EXPECT_FLOAT_EQ(config2.sigmaX, 7);
}

TEST_F(ConfigTest, ConfigWriteToJSONRoundTrip) {
    DeconvolutionConfig config1;
    config1.algorithmName = "Convolution";
    config1.iterations = 3;
    config1.epsilon = 1e-3f;
    json j = config1.writeToJSON();

    DeconvolutionConfig config2;
    ASSERT_TRUE(config2.loadFromJSON(j));
    EXPECT_EQ(config2.algorithmName, "Convolution");
    EXPECT_EQ(config2.iterations, 3);
    EXPECT_FLOAT_EQ(config2.epsilon, 1e-3f);
}
