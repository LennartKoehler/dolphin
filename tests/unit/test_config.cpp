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
#include "dolphin/Config.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"
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

    // Deconvolution configs are one class per algorithm; the factory builds the
    // one registered for `algorithm`.
    static std::shared_ptr<DeconvolutionConfig> makeDeconvConfig(const std::string& algorithm) {
        return DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm(algorithm);
    }
};


TEST_F(ConfigTest, DeconvolutionConfigLoadFromJSON) {
    auto jsonStr = TestUtils::defaultDeconvConfigJSON();
    json j = json::parse(jsonStr);
    auto config = DeconvolutionConfig::createFromJSON(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->algorithmName, "RichardsonLucy");
    EXPECT_EQ(config->iterations, 10);
    EXPECT_FLOAT_EQ(config->minDenominator, 1e-6f);
    EXPECT_FLOAT_EQ(config->signalToNoise, 0.001f);
}

TEST_F(ConfigTest, DeconvolutionConfigWriteToJSON) {
    auto config = makeDeconvConfig("InverseFilter");
    config->iterations = 50;
    json j = config->writeToJSON();
    ASSERT_TRUE(j.contains("algorithm_name"));
    EXPECT_EQ(j["algorithm_name"], "InverseFilter");
    EXPECT_EQ(j["iterations"], 50);
}

TEST_F(ConfigTest, DeconvolutionConfigRoundTrip) {
    auto config1 = makeDeconvConfig("RichardsonLucyTotalVariation");
    config1->iterations = 25;
    config1->signalToNoise = 0.05f;
    json j = config1->writeToJSON();

    auto config2 = DeconvolutionConfig::createFromJSON(j);
    ASSERT_NE(config2, nullptr);
    EXPECT_EQ(config2->algorithmName, "RichardsonLucyTotalVariation");
    EXPECT_EQ(config2->iterations, 25);
    EXPECT_FLOAT_EQ(config2->signalToNoise, 0.05f);
}

TEST_F(ConfigTest, PoissonMLEConfigRoundTrip) {
    auto config1 = makeDeconvConfig("PoissonMLE");
    auto* poisson1 = dynamic_cast<PoissonMLEConfig*>(config1.get());
    ASSERT_NE(poisson1, nullptr);
    poisson1->iterations = 12;
    poisson1->poissonSieve = false;
    poisson1->poissonBackground = 4.5f;
    poisson1->poissonSievePasses = 5;

    json j = config1->writeToJSON();
    ASSERT_TRUE(j.contains("poisson_sieve"));
    ASSERT_TRUE(j.contains("poisson_background"));

    auto config2 = DeconvolutionConfig::createFromJSON(j);
    auto* poisson2 = dynamic_cast<PoissonMLEConfig*>(config2.get());
    ASSERT_NE(poisson2, nullptr);
    EXPECT_EQ(config2->algorithmName, "PoissonMLE");
    EXPECT_EQ(poisson2->iterations, 12);
    EXPECT_FALSE(poisson2->poissonSieve);
    EXPECT_FLOAT_EQ(poisson2->poissonBackground, 4.5f);
    EXPECT_EQ(poisson2->poissonSievePasses, 5);
}

TEST_F(ConfigTest, AlgorithmSpecificParametersOnlyRegisteredForTheirAlgorithm) {
    auto rl = makeDeconvConfig("RichardsonLucy");
    json rlJson = rl->writeToJSON();
    EXPECT_FALSE(rlJson.contains("poisson_sieve"));
    EXPECT_FALSE(rlJson.contains("poisson_background"));

    auto poisson = makeDeconvConfig("PoissonMLE");
    json poissonJson = poisson->writeToJSON();
    EXPECT_TRUE(poissonJson.contains("poisson_sieve"));
    EXPECT_TRUE(poissonJson.contains("poisson_background"));
}

TEST_F(ConfigTest, UnknownAlgorithmInJSONThrows) {
    json j = {{"algorithm_name", "NonExistent"}};
    EXPECT_THROW(DeconvolutionConfig::createFromJSON(j), std::runtime_error);
}

TEST_F(ConfigTest, DeconvolutionConfigCopyConstructor) {
    // Copies must re-register parameters: the void* in Config::parameters would
    // dangle otherwise. Check both base and derived fields survive, independently.
    auto original = std::make_shared<PoissonMLEConfig>();
    original->iterations = 42;
    original->paddingStrategyType = PaddingStrategyType::LOW;
    original->poissonSievePasses = 9;

    PoissonMLEConfig copy(*original);
    EXPECT_EQ(copy.algorithmName, "PoissonMLE");
    EXPECT_EQ(copy.iterations, 42);
    EXPECT_EQ(copy.paddingStrategyType, PaddingStrategyType::LOW);
    EXPECT_EQ(copy.poissonSievePasses, 9);

    copy.iterations = 43;
    copy.poissonSievePasses = 10;
    EXPECT_EQ(original->iterations, 42);
    EXPECT_EQ(original->poissonSievePasses, 9);

    // The copy's registered parameters point at the copy, not the original.
    json written = copy.writeToJSON();
    EXPECT_EQ(written["iterations"], 43);
    EXPECT_EQ(written["poisson_sieve_passes"], 10);
}

TEST_F(ConfigTest, DeconvolutionConfigAssignment) {
    auto config1 = makeDeconvConfig("RichardsonLucy");
    config1->iterations = 99;
    auto config2 = config1;
    EXPECT_EQ(config2->iterations, 99);
}

TEST_F(ConfigTest, PoissonMLEConfigCopyIsIndependent) {
    auto original = makeDeconvConfig("PoissonMLE");
    auto* poisson = dynamic_cast<PoissonMLEConfig*>(original.get());
    ASSERT_NE(poisson, nullptr);
    poisson->poissonSievePasses = 9;

    auto copied = DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm("PoissonMLE");
    *static_cast<PoissonMLEConfig*>(copied.get()) = *poisson;
    auto* copiedPoisson = dynamic_cast<PoissonMLEConfig*>(copied.get());
    ASSERT_NE(copiedPoisson, nullptr);
    EXPECT_EQ(copiedPoisson->poissonSievePasses, 9);
    copiedPoisson->poissonSievePasses = 1;
    EXPECT_EQ(poisson->poissonSievePasses, 9);
}

TEST_F(ConfigTest, DeconvolutionConfigCreateFromJSONFile) {
    auto jsonStr = TestUtils::defaultDeconvConfigJSON();
    auto filePath = TestUtils::outputPath() + "/deconv_config.json";
    std::ofstream file(filePath);
    file << jsonStr;
    file.close();

    auto config = DeconvolutionConfig::createFromJSONFile(filePath);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->algorithmName, "RichardsonLucy");
    EXPECT_EQ(config->iterations, 10);
}

TEST_F(ConfigTest, DeconvolutionConfigMissingFile) {
    EXPECT_THROW(DeconvolutionConfig::createFromJSONFile("/nonexistent/path.json"), std::exception);
}

TEST_F(ConfigTest, SetupConfigDefaults) {
    SetupConfig config;
    EXPECT_EQ(config.backend, "cpu");
    // 0 means "auto": resolved by the deconvolution strategy / backend manager.
    EXPECT_EQ(config.threadConfig.nThreads, 0u);
    EXPECT_EQ(config.threadConfig.nIOThreads, 0u);
    EXPECT_EQ(config.threadConfig.nWorkerThreads, 0u);
    EXPECT_EQ(config.nDevices, 0u);
    EXPECT_EQ(config.savePsf, false);
}

TEST_F(ConfigTest, SetupConfigLoadFromJSON) {
    auto jsonStr = TestUtils::defaultSetupConfigJSON();
    json j = json::parse(jsonStr);
    SetupConfig config;
    ASSERT_TRUE(config.loadFromJSON(j));
    EXPECT_EQ(config.backend, "cpu");
    EXPECT_EQ(config.threadConfig.nIOThreads, 1);
    EXPECT_EQ(config.threadConfig.nWorkerThreads, 1);
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
    config1.threadConfig.nThreads = 8;
    SetupConfig config2(config1);
    EXPECT_EQ(config2.backend, "cuda");
    EXPECT_EQ(config2.threadConfig.nThreads, 8);
}

TEST_F(ConfigTest, SetupConfigAssignment) {
    SetupConfig config1;
    config1.threadConfig.nIOThreads = 5;
    SetupConfig config2;
    config2 = config1;
    EXPECT_EQ(config2.threadConfig.nIOThreads, 5);
}

TEST_F(ConfigTest, GaussianPSFConfigDefaults) {
    GaussianPSFConfig config;
    EXPECT_EQ(config.psfModelName, "Gaussian");
    EXPECT_EQ(config.sizeX, 0);
    EXPECT_EQ(config.sizeY, 0);
    EXPECT_EQ(config.sizeZ, 0);
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
    EXPECT_EQ(config->sizeX, 0);
    EXPECT_EQ(config->sizeY, 0);
    EXPECT_EQ(config->sizeZ, 0);

    auto* glConfig = dynamic_cast<GibsonLanniPSFConfig*>(config.get());
    ASSERT_NE(glConfig, nullptr);
    EXPECT_FLOAT_EQ(glConfig->NA, 1.4f);
    EXPECT_FLOAT_EQ(glConfig->wavelength_nm, 450.0f);
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
    auto config1 = makeDeconvConfig("Convolution");
    config1->iterations = 3;
    config1->minDenominator = 1e-3f;
    json j = config1->writeToJSON();

    auto config2 = DeconvolutionConfig::createFromJSON(j);
    ASSERT_NE(config2, nullptr);
    EXPECT_EQ(config2->algorithmName, "Convolution");
    EXPECT_EQ(config2->iterations, 3);
    EXPECT_FLOAT_EQ(config2->minDenominator, 1e-3f);
}
