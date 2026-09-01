#include <gtest/gtest.h>
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/generators/GibsonLanniPSFGenerator.h"
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
    CuboidShape shape = psf.getShape();
    EXPECT_GT(shape.width, 0u);
    EXPECT_GT(shape.height, 0u);
    EXPECT_GT(shape.depth, 0u);
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

    CuboidShape shape = psf.getShape();
    float maxVal = psf.getMax();
    float centerVal = psf.getPixel(shape.width / 2, shape.height / 2, shape.depth / 2);
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

// --- Fixed-size PSF generation tests ---

TEST_F(PSFGeneratorTest, GibsonLanniFixedSize) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["size_x"] = 33;
    j["size_y"] = 33;
    j["size_z"] = 17;
    auto generator = factory.createGenerator("GibsonLanni", j);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, 33u);
    EXPECT_EQ(shape.height, 33u);
    EXPECT_EQ(shape.depth, 17u);

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

// --- Threshold-based PSF cutoff tests ---

TEST_F(PSFGeneratorTest, GibsonLanniThresholdCutoff) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_LT(shape.width, 64u);
    EXPECT_LT(shape.height, 64u);
    EXPECT_LT(shape.depth, 32u);

    float maxVal = psf.getMax();
    float edgeVal = psf.getPixel(0, 0, 0);
    EXPECT_LT(edgeVal, maxVal * 5e-2f);

    float centerEdgeZ = psf.getPixel(shape.width / 2, shape.height / 2, 0);
    EXPECT_LT(centerEdgeZ, maxVal * 5e-2f);
}

TEST_F(PSFGeneratorTest, GibsonLanniThresholdPeakCentered) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["particle_axial_position_nm"] = 1000.0f;
    auto generator = factory.createGenerator("GibsonLanni", j);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    size_t cx = shape.width / 2;
    size_t cy = shape.height / 2;
    size_t cz = shape.depth / 2;

    float maxVal = 0.0f;
    size_t maxZ = 0;
    for (size_t z = 0; z < shape.depth; z++) {
        float val = psf.getPixel(cx, cy, z);
        if (val > maxVal) { maxVal = val; maxZ = z; }
    }

    EXPECT_LE(std::abs(static_cast<long>(maxZ) - static_cast<long>(cz)), 2);
}

TEST_F(PSFGeneratorTest, GibsonLanniNoNaN) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto generator = factory.createGenerator("GibsonLanni", j);
    PSF psf = generator->generatePSF();

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

// --- Energy-based PSF extent tests ---

TEST_F(PSFGeneratorTest, EnergyExtentGibsonLanniAxial) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["particle_axial_position_nm"] = 1000.0f;
    auto generator = factory.createGenerator("GibsonLanni", j);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    size_t cx = shape.width / 2;
    size_t cy = shape.height / 2;

    float maxVal = 0.0f;
    size_t zPeak = 0;
    for (size_t z = 0; z < shape.depth; z++) {
        float val = psf.getPixel(cx, cy, z);
        if (val > maxVal) { maxVal = val; zPeak = z; }
    }

    PSFExtent extent = psf.computeEnergyExtent(1.0, 0.90);

    double axialTotal = 0.0;
    for (size_t z = 0; z < shape.depth; z++) {
        axialTotal += psf.getPixel(cx, cy, z);
    }

    auto axialEnergyAt = [&](size_t d) -> double {
        double sum = 0.0;
        long lo = static_cast<long>(zPeak) - static_cast<long>(d);
        long hi = static_cast<long>(zPeak) + static_cast<long>(d);
        for (size_t z = 0; z < shape.depth; z++) {
            if (static_cast<long>(z) >= lo && static_cast<long>(z) <= hi)
                sum += psf.getPixel(cx, cy, z);
        }
        return sum / axialTotal;
    };

    EXPECT_GE(axialEnergyAt(extent.zHalfExtent), 0.90);
    if (extent.zHalfExtent > 0) {
        EXPECT_LT(axialEnergyAt(extent.zHalfExtent - 1), 0.90);
    }
}

TEST_F(PSFGeneratorTest, EnergyExtentGibsonLanniLateral) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto generator = factory.createGenerator("GibsonLanni", j);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    size_t cx = shape.width / 2;
    size_t cy = shape.height / 2;

    float maxVal = 0.0f;
    size_t zPeak = 0;
    for (size_t z = 0; z < shape.depth; z++) {
        float val = psf.getPixel(cx, cy, z);
        if (val > maxVal) { maxVal = val; zPeak = z; }
    }

    PSFExtent extent = psf.computeEnergyExtent(0.90, 1.0);

    double lateralTotal = 0.0;
    for (size_t x = 0; x < shape.width; x++) {
        lateralTotal += psf.getPixel(x, cy, zPeak);
    }

    auto lateralEnergyAt = [&](size_t r) -> double {
        double sum = 0.0;
        for (size_t x = 0; x < shape.width; x++) {
            if (static_cast<long>(x) >= static_cast<long>(cx) - static_cast<long>(r) &&
                static_cast<long>(x) <= static_cast<long>(cx) + static_cast<long>(r))
                sum += psf.getPixel(x, cy, zPeak);
        }
        return sum / lateralTotal;
    };

    EXPECT_GE(lateralEnergyAt(extent.lateralExtent), 0.90);
    if (extent.lateralExtent > 0) {
        EXPECT_LT(lateralEnergyAt(extent.lateralExtent - 1), 0.90);
    }
}

TEST_F(PSFGeneratorTest, EnergyExtentGaussian) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto generator = factory.createGenerator("Gaussian", j);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    size_t cx = shape.width / 2;
    size_t cy = shape.height / 2;
    size_t cz = shape.depth / 2;

    double axialTotal = 0.0;
    for (size_t z = 0; z < shape.depth; z++) {
        axialTotal += psf.getPixel(cx, cy, z);
    }

    PSFExtent extent = psf.computeEnergyExtent(0.90, 0.90);

    auto axialEnergyAt = [&](size_t d) -> double {
        double sum = 0.0;
        for (size_t z = 0; z < shape.depth; z++) {
            if (z >= cz - d && z <= cz + d)
                sum += psf.getPixel(cx, cy, z);
        }
        return sum / axialTotal;
    };

    EXPECT_GE(axialEnergyAt(extent.zHalfExtent), 0.90);
    if (extent.zHalfExtent > 0) {
        EXPECT_LT(axialEnergyAt(extent.zHalfExtent - 1), 0.90);
    }
}
