#include <gtest/gtest.h>
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"
#include "nlohmann/json.hpp"
#include <cmath>

using json = nlohmann::json;

class PSFGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(PSFGeneratorTest, FactoryGetAvailableModels) {
    auto& factory = PSFGeneratorFactory::getInstance();
    auto models = factory.getAvailablePSFModels();
    EXPECT_GE(models.size(), 2u);
}

TEST_F(PSFGeneratorTest, FactoryHasGaussian) {
    auto& factory = PSFGeneratorFactory::getInstance();
    auto models = factory.getAvailablePSFModels();
    bool found = false;
    for (const auto& m : models) {
        if (m == "Gaussian") found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(PSFGeneratorTest, FactoryHasGibsonLanni) {
    auto& factory = PSFGeneratorFactory::getInstance();
    auto models = factory.getAvailablePSFModels();
    bool found = false;
    for (const auto& m : models) {
        if (m == "GibsonLanni") found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(PSFGeneratorTest, CreateGaussianConfig) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto config = factory.createConfig(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModelName(), "Gaussian");
    EXPECT_EQ(config->sizeX, 32);
}

TEST_F(PSFGeneratorTest, CreateGibsonLanniConfig) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto config = factory.createConfig(j);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModelName(), "GibsonLanni");
}

TEST_F(PSFGeneratorTest, CreateConfigInvalidModel) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = {{"model_name", "NonExistent"}, {"size_x", 10}, {"size_y", 10}, {"size_z", 10}};
    EXPECT_THROW(factory.createConfig(j), std::runtime_error);
}

TEST_F(PSFGeneratorTest, CreateConfigMissingModelName) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = {{"size_x", 10}, {"size_y", 10}, {"size_z", 10}};
    EXPECT_THROW(factory.createConfig(j), std::runtime_error);
}

TEST_F(PSFGeneratorTest, GaussianPSFGeneration) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);
    ASSERT_NE(generator, nullptr);
    EXPECT_TRUE(generator->hasConfig());

    PSF psf = generator->generatePSF();
    EXPECT_EQ(psf.getShape(), CuboidShape(32, 32, 16));
    EXPECT_FALSE(psf.ID.empty());
}

TEST_F(PSFGeneratorTest, GaussianPSFIsCentered) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);
    PSF psf = generator->generatePSF();

    float maxVal = psf.getMax();
    float centerVal = psf.getPixel(32 / 2, 32 / 2, 16 / 2);
    EXPECT_NEAR(centerVal, maxVal, maxVal * 0.01f);
}

TEST_F(PSFGeneratorTest, GaussianPSFSymmetric) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);
    PSF psf = generator->generatePSF();

    int cx = 32 / 2;
    int cy = 32 / 2;
    int cz = 16 / 2;

    for (int dz = -3; dz <= 3; dz++) {
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                float v1 = psf.getPixel(cx + dx, cy + dy, cz + dz);
                float v2 = psf.getPixel(cx - 1 - dx, cy - 1 - dy, cz - 1 - dz);
                EXPECT_NEAR(v1, v2, v1 * 0.01f + 1e-6f);
            }
        }
    }
}

TEST_F(PSFGeneratorTest, GaussianPSFNoNaN) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);
    PSF psf = generator->generatePSF();

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GaussianPSFGetPadding) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);
    auto padding = generator->getPadding(PaddingStrategyType::PARENT);
    EXPECT_GE(padding.width, 0);
    EXPECT_GE(padding.height, 0);
    EXPECT_GE(padding.depth, 0);
}

TEST_F(PSFGeneratorTest, GaussianPSFCreateFromConfig) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto config = factory.createConfig(j);
    auto generator = factory.createGenerator(config);
    ASSERT_NE(generator, nullptr);
    EXPECT_TRUE(generator->hasConfig());

    PSF psf = generator->generatePSF();
    EXPECT_EQ(psf.getShape(), CuboidShape(32, 32, 16));
}

TEST_F(PSFGeneratorTest, GibsonLanniPSFGeneration) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto generator = factory.createGenerator("GibsonLanni", j);
    ASSERT_NE(generator, nullptr);
    EXPECT_TRUE(generator->hasConfig());

    PSF psf = generator->generatePSF();
    EXPECT_EQ(psf.getShape(), CuboidShape(64, 64, 32));
}

TEST_F(PSFGeneratorTest, GibsonLanniPSFNoNaN) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto generator = factory.createGenerator("GibsonLanni", j);
    PSF psf = generator->generatePSF();

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniPSFCentered) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto generator = factory.createGenerator("GibsonLanni", j);
    PSF psf = generator->generatePSF();

    float maxVal = psf.getMax();
    float centerVal = psf.getPixel(64 / 2, 64 / 2, 32 / 2);
    EXPECT_NEAR(centerVal, maxVal, maxVal * 0.1f);
}

TEST_F(PSFGeneratorTest, PSFWriteAndReadTiff) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);
    PSF psf = generator->generatePSF();

    auto path = TestUtils::outputPath() + "/psf_test.tif";
    psf.writeToTiffFile(path);

    PSF readPSF;
    readPSF.readFromTiffFile(path);

    EXPECT_EQ(readPSF.getShape(), psf.getShape());
    EXPECT_TRUE(readPSF.isEqual(psf, 0.001f));
}

TEST_F(PSFGeneratorTest, PSFConstructorWithID) {
    Image3D img(CuboidShape(4, 4, 4), 1.0f);
    PSF psf(std::move(img), "my_id");
    EXPECT_EQ(psf.ID, "my_id");
    EXPECT_EQ(psf.getShape(), CuboidShape(4, 4, 4));
}
