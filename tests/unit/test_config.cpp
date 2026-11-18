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
#include "dolphin/psf/configs/ModalityConfig.h"
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


TEST_F(ConfigTest, DeconvolutionConfigLoadFromJSON) {
    auto jsonStr = TestUtils::defaultDeconvConfigJSON();
    json j = json::parse(jsonStr);
    DeconvolutionConfig config;
    ASSERT_TRUE(config.loadFromJSON(j));
    EXPECT_EQ(config.algorithmName, "RichardsonLucy");
    EXPECT_EQ(config.iterations, 10);
    EXPECT_FLOAT_EQ(config.min_denominator, 1e-6f);
    EXPECT_FLOAT_EQ(config.signal_to_noise, 0.001f);
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
    config1.signal_to_noise = 0.05f;
    json j = config1.writeToJSON();

    DeconvolutionConfig config2;
    ASSERT_TRUE(config2.loadFromJSON(j));
    EXPECT_EQ(config2.algorithmName, "RichardsonLucyTotalVariation");
    EXPECT_EQ(config2.iterations, 25);
    EXPECT_FLOAT_EQ(config2.signal_to_noise, 0.05f);
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

TEST_F(ConfigTest, PSFConfigMicroscopeModalityDefaultsWidefield) {
    auto config = PSFConfig::createFromJSON(json::parse(TestUtils::gibsonLanniPSFConfigJSON()));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModalityType(), PSFConfig::MicroscopeModality::WIDEFIELD);
    EXPECT_EQ(config->getModalityName(), "widefield");
    // The modality config object is a widefield instance by default.
    EXPECT_NE(config->modality<WidefieldModalityConfig>(), nullptr);
    EXPECT_EQ(config->modality<ConfocalModalityConfig>(), nullptr);
}

TEST_F(ConfigTest, PSFConfigMicroscopeModalityFromJSON) {
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["microscope_modality"] = "confocal";
    j["excitation_wavelength_nm"] = 488.0;
    j["pinhole_radius_nm"] = 280.0;
    auto config = PSFConfig::createFromJSON(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModalityType(), PSFConfig::MicroscopeModality::CONFOCAL);
    EXPECT_EQ(config->getModalityName(), "confocal");
    // The modality config object carries the modality's parameters.
    const ConfocalModalityConfig* modalityCfg = config->modality<ConfocalModalityConfig>();
    ASSERT_NE(modalityCfg, nullptr);
    EXPECT_FLOAT_EQ(modalityCfg->excitationWavelength_nm, 488.0f);
    EXPECT_FLOAT_EQ(modalityCfg->pinholeRadius_nm, 280.0f);

    j["microscope_modality"] = "widefield";
    EXPECT_EQ(PSFConfig::createFromJSON(j)->getModalityName(), "widefield");
}

TEST_F(ConfigTest, PSFConfigMicroscopeModalityInvalidFallsBack) {
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["microscope_modality"] = "lightfield";
    auto config = PSFConfig::createFromJSON(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModalityType(), PSFConfig::MicroscopeModality::WIDEFIELD);
}

TEST_F(ConfigTest, PSFConfigMicroscopeModalityCopies) {
    auto config = std::make_shared<GibsonLanniPSFConfig>();
    config->selectModality(MicroscopeModality::CONFOCAL);
    // The generator copies the config via the GibsonLanni copy constructor;
    // the modality config object must be cloned with it.
    GibsonLanniPSFConfig copied(*config);
    EXPECT_EQ(copied.getModalityName(), "confocal");
    EXPECT_NE(copied.modalityConfig, config->modalityConfig);
    // The modality config object (source of truth) is the confocal type.
    EXPECT_EQ(copied.getModalityType(), MicroscopeModality::CONFOCAL);
}

TEST_F(ConfigTest, WavelengthKeysOwnedByTheirConfig) {
    // Widefield: the optical model's "wavelength_nm" is the detection
    // wavelength; the confocal keys do not exist on a widefield config.
    json widefield = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    widefield["wavelength_nm"] = 540.0;
    auto widefieldConfig = PSFConfig::createFromJSON(widefield);
    auto* glWidefield = dynamic_cast<GibsonLanniPSFConfig*>(widefieldConfig.get());
    ASSERT_NE(glWidefield, nullptr);
    EXPECT_FLOAT_EQ(glWidefield->wavelength_nm, 540.0f);
    EXPECT_EQ(glWidefield->modality<ConfocalModalityConfig>(), nullptr);

    // Confocal: emission and excitation wavelengths live on the modality
    // config, under their own keys.
    json confocal = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    confocal["microscope_modality"] = "confocal";
    confocal["emission_wavelength_nm"] = 530.0;
    confocal["excitation_wavelength_nm"] = 488.0;
    auto confocalConfig = PSFConfig::createFromJSON(confocal);
    auto* gl = dynamic_cast<GibsonLanniPSFConfig*>(confocalConfig.get());
    ASSERT_NE(gl, nullptr);
    const ConfocalModalityConfig* modalityCfg = gl->modality<ConfocalModalityConfig>();
    ASSERT_NE(modalityCfg, nullptr);
    EXPECT_FLOAT_EQ(modalityCfg->emissionWavelength_nm, 530.0f);
    EXPECT_FLOAT_EQ(modalityCfg->excitationWavelength_nm, 488.0f);

    // Confocal: "wavelength_nm" fills the (unused) optical-model parameter
    // but does not touch the confocal emission wavelength.
    json stray = confocal;
    stray["wavelength_nm"] = 560.0;
    auto strayConfig = PSFConfig::createFromJSON(stray);
    auto* glStray = dynamic_cast<GibsonLanniPSFConfig*>(strayConfig.get());
    ASSERT_NE(glStray, nullptr);
    EXPECT_FLOAT_EQ(glStray->wavelength_nm, 560.0f);
    EXPECT_FLOAT_EQ(glStray->modality<ConfocalModalityConfig>()->emissionWavelength_nm, 530.0f);
}

TEST_F(ConfigTest, WavelengthRoundTrip) {
    // Confocal configs save the confocal wavelengths under the modality's
    // keys; widefield configs keep the optical model's "wavelength_nm".
    json confocal = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    confocal["microscope_modality"] = "confocal";
    confocal["emission_wavelength_nm"] = 530.0;
    confocal["excitation_wavelength_nm"] = 488.0;
    json confocalOut = PSFConfig::createFromJSON(confocal)->writeToJSON();
    EXPECT_TRUE(confocalOut.contains("emission_wavelength_nm"));
    EXPECT_TRUE(confocalOut.contains("excitation_wavelength_nm"));
    EXPECT_FLOAT_EQ(confocalOut.at("emission_wavelength_nm").get<float>(), 530.0f);
    // Round-trip: the saved config reloads with the same values.
    auto reloaded = PSFConfig::createFromJSON(confocalOut);
    auto* glReloaded = dynamic_cast<GibsonLanniPSFConfig*>(reloaded.get());
    ASSERT_NE(glReloaded, nullptr);
    ASSERT_NE(glReloaded, nullptr);
    const ConfocalModalityConfig* modalityCfg = glReloaded->modality<ConfocalModalityConfig>();
    ASSERT_NE(modalityCfg, nullptr);
    EXPECT_FLOAT_EQ(modalityCfg->emissionWavelength_nm, 530.0f);
    EXPECT_FLOAT_EQ(modalityCfg->excitationWavelength_nm, 488.0f);

    json widefieldOut = PSFConfig::createFromJSON(json::parse(TestUtils::gibsonLanniPSFConfigJSON()))->writeToJSON();
    EXPECT_TRUE(widefieldOut.contains("wavelength_nm"));
    EXPECT_FALSE(widefieldOut.contains("emission_wavelength_nm"));
    // The modality is written back from the config object itself: it is not a
    // registry parameter, so it has to be added explicitly on save.
    EXPECT_EQ(widefieldOut.at("microscope_modality").get<std::string>(), "widefield");
    EXPECT_EQ(confocalOut.at("microscope_modality").get<std::string>(), "confocal");
}

TEST_F(ConfigTest, MicroscopeModalityTagRoundTripsWithoutShadowMember) {
    // The modality config object is the only source of truth: there is no enum
    // member to desync. Loading, switching in memory and saving must agree.
    auto config = std::make_shared<GibsonLanniPSFConfig>();
    EXPECT_EQ(config->getModalityName(), "widefield");

    // In-memory switch (no JSON involved) is reflected on save.
    config->selectModality(MicroscopeModality::CONFOCAL);
    EXPECT_EQ(config->writeToJSON().at("microscope_modality").get<std::string>(), "confocal");

    // A copy made from that state keeps the modality and its parameters.
    GibsonLanniPSFConfig copied(*config);
    EXPECT_EQ(copied.getModalityType(), MicroscopeModality::CONFOCAL);
    const ConfocalModalityConfig* modalityCfg = copied.modality<ConfocalModalityConfig>();
    ASSERT_NE(modalityCfg, nullptr);
    EXPECT_EQ(copied.writeToJSON().at("microscope_modality").get<std::string>(), "confocal");

    // Switching back is equally reflected.
    config->selectModality(MicroscopeModality::WIDEFIELD);
    EXPECT_EQ(config->modality<ConfocalModalityConfig>(), nullptr);
    EXPECT_EQ(config->writeToJSON().at("microscope_modality").get<std::string>(), "widefield");
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
    config1.min_denominator = 1e-3f;
    json j = config1.writeToJSON();

    DeconvolutionConfig config2;
    ASSERT_TRUE(config2.loadFromJSON(j));
    EXPECT_EQ(config2.algorithmName, "Convolution");
    EXPECT_EQ(config2.iterations, 3);
    EXPECT_FLOAT_EQ(config2.min_denominator, 1e-3f);
}
