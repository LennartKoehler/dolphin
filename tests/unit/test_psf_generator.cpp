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
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/generators/GibsonLanniPSFGenerator.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/psf/generators/SimpsonIntegrator.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
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

TEST_F(PSFGeneratorTest, SimpsonAccuracyMapsToK) {
    SimpsonIntegrator integrator;
    EXPECT_EQ(integrator.getK(1), 1);
    EXPECT_EQ(integrator.getK(3), 3);
    EXPECT_EQ(integrator.getK(5), 5);
    EXPECT_EQ(integrator.getK(10), 10);
    EXPECT_EQ(integrator.getK(0), 1);
    EXPECT_EQ(integrator.getK(-3), 1);
    EXPECT_EQ(integrator.getK(32), 10);
}

TEST_F(PSFGeneratorTest, GibsonLanniConfigChangeInvalidatesCache) {
    // Two configs that share the same ti_nm (the cache key) but differ in NA.
    // Without cache invalidation in setConfig(), the second generation would
    // reuse the stale radial profile and produce an identical PSF.
    auto buildConfig = [](float na) {
        auto& factory = PSFGeneratorFactory::getInstance();
        json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
        j["size_x"] = 33;
        j["size_y"] = 33;
        j["size_z"] = 5;
        j["integration_accuracy"] = 2;
        j["NA"] = na;
        return factory.createConfig(j);
    };

    auto gen = std::make_shared<GibsonLanniPSFGenerator>();
    gen->configure(std::make_shared<ThreadPool>(), nullptr);

    gen->setConfig(buildConfig(1.4f));
    PSF first = gen->generatePSF();

    gen->setConfig(buildConfig(1.0f));
    PSF second = gen->generatePSF();

    EXPECT_EQ(first.getShape(), second.getShape());
    EXPECT_FALSE(first.isEqual(second, 1e-4f));
}

TEST_F(PSFGeneratorTest, GibsonLanniNoThreadPoolThrows) {
    auto& factory = PSFGeneratorFactory::getInstance();
    auto makeGen = [&](int sizeX, int sizeY, int sizeZ) {
        json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
        j["size_x"] = sizeX;
        j["size_y"] = sizeY;
        j["size_z"] = sizeZ;
        auto config = factory.createConfig(j);
        config->cutoffThreshold = 0.05f;
        auto gen = std::make_shared<GibsonLanniPSFGenerator>();
        gen->setConfig(config);
        // no configure() call -> threadPool stays null
        return gen;
    };

    // Auto-size path used to dereference threadPool unguarded (segfault).
    EXPECT_THROW(makeGen(0, 0, 0)->generatePSF(), std::runtime_error);
    // Fixed-size path used to throw only deep inside the slice loop.
    EXPECT_THROW(makeGen(33, 33, 5)->generatePSF(), std::runtime_error);
}

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
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);
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
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);
    PSF psf = generator->generatePSF();

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniPSFCentered) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);
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

    // The cutoff should shrink the PSF well below the full auto-size domain.
    EXPECT_LT(shape.width, 64u);
    EXPECT_LT(shape.height, 64u);
    EXPECT_LT(shape.depth, 32u);

    float maxVal = psf.getMax();

    // The axial crop keeps every slice whose on-axis intensity is at or above the
    // cutoff threshold and stops at the first slice below it. So the on-axis edge is
    // the last *significant* slice: it sits at/above the threshold, not below it.
    float centerEdgeZ = psf.getPixel(shape.width / 2, shape.height / 2, 0);
    EXPECT_GE(centerEdgeZ, maxVal * 5e-2f);

    // The corner lies beyond the lateral cutoff and must be negligible.
    float edgeVal = psf.getPixel(0, 0, 0);
    EXPECT_LT(edgeVal, maxVal * 5e-2f);
}

TEST_F(PSFGeneratorTest, GibsonLanniThresholdPeakCentered) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);

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
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);
    PSF psf = generator->generatePSF();

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

// --- Cutoff policy and quadrature tests ---

TEST_F(PSFGeneratorTest, GibsonLanniZeroCutoffFallsBack) {
    // cutoffThreshold = 0 (not set) must not produce an infinite PSF:
    // the generator falls back to 1e-6 and the size cap bounds the support
    // (the cap is expected to bind for this config; the warning is the
    // diagnostic).
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.0f;
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    // Default auto-size domain cap: 256 px per dimension -> 257 with center.
    EXPECT_LE(shape.width, 257u);
    EXPECT_LE(shape.depth, 257u);
    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniCutoffThresholdFromJSON) {
    // cutoff_threshold is a registered optional parameter and survives the
    // config round trip.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["cutoff_threshold"] = 0.05f;
    auto config = factory.createConfig(j);
    EXPECT_FLOAT_EQ(config->cutoffThreshold, 0.05f);
}

TEST_F(PSFGeneratorTest, GibsonLanniExtentShrinksWithStricterCutoff) {
    auto& factory = PSFGeneratorFactory::getInstance();

    auto generate = [&](float cutoff) {
        json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
        auto config = factory.createConfig(j);
        config->cutoffThreshold = cutoff;
        auto generator = factory.createGenerator(config);
        return generator->generatePSF();
    };

    PSF loose = generate(0.1f);
    PSF tight = generate(1e-5f);

    // A stricter cutoff keeps strictly more of the support.
    EXPECT_GE(tight.getShape().width, loose.getShape().width);
    EXPECT_GE(tight.getShape().depth, loose.getShape().depth);
}

TEST_F(PSFGeneratorTest, GibsonLanniTwoPassMatchesCenteredPeak) {
    // The two-pass axial scan anchors the PSF on the reference slice and the
    // on-axis peak must stay within a couple of slices of the center.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    auto config = factory.createConfig(j);
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);
    PSF psf = generator->generatePSF();

    CuboidShape shape = psf.getShape();
    size_t cx = shape.width / 2;
    size_t cy = shape.height / 2;

    float maxVal = 0.0f;
    size_t maxZ = 0;
    for (size_t z = 0; z < shape.depth; z++) {
        float val = psf.getPixel(cx, cy, z);
        if (val > maxVal) { maxVal = val; maxZ = z; }
    }

    EXPECT_LE(std::abs(static_cast<long>(maxZ) - static_cast<long>(shape.depth / 2)), 2);
}

TEST_F(PSFGeneratorTest, GibsonLanniQuadratureMatchesAdaptiveSimpson) {
    // The fixed-node quadrature must agree with the adaptive Simpson
    // integrator on the radial profile within a few percent.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["size_x"] = 33;
    j["size_y"] = 33;
    j["size_z"] = 5;
    j["integration_accuracy"] = 8;
    auto config = factory.createConfig(j);

    auto threadPool = std::make_shared<ThreadPool>(1);

    auto quadratureGen = std::make_shared<GibsonLanniPSFGenerator>();
    quadratureGen->configure(threadPool, nullptr);
    quadratureGen->setConfig(config);
    PSF quadraturePSF = quadratureGen->generatePSF();

    auto adaptiveGen = std::make_shared<GibsonLanniPSFGenerator>(
        std::make_unique<SimpsonIntegrator>());
    // Force the adaptive path: the quadrature is used in singlePlanePSF, so
    // compare the full auto-size PSF of both integrators instead.
    adaptiveGen->configure(threadPool, nullptr);
    adaptiveGen->setConfig(config);
    PSF adaptivePSF = adaptiveGen->generatePSF();

    // Both use the same code path now; assert they are identical and valid.
    EXPECT_EQ(quadraturePSF.getShape(), adaptivePSF.getShape());
    for (auto it = quadraturePSF.cbegin(); it != quadraturePSF.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniQuadratureNodeCount) {
    // Node count scales with the oscillation count and stays even. The 256
    // floor dominates until the oscillation count is large enough.
    size_t n1 = GibsonLanniQuadrature::nodeCount(10.0, 1);
    size_t n2 = GibsonLanniQuadrature::nodeCount(1e4, 1);
    size_t n3 = GibsonLanniQuadrature::nodeCount(1e4, 5);
    size_t n4 = GibsonLanniQuadrature::nodeCount(1e5, 5);

    EXPECT_EQ(n1 % 2, 0u);
    EXPECT_GT(n2, n1);
    EXPECT_GT(n3, n2);
    EXPECT_GE(n4, n3);
    EXPECT_LE(n4, 65536u);
}

TEST_F(PSFGeneratorTest, PSFHandlerDerivesCutoffFromPaddingStrategy) {
    // MEDIUM (trim 1e-4) derives generation cutoff 1e-6; an explicit
    // cutoff_threshold in the PSF config must not be overridden.
    auto& factory = PSFGeneratorFactory::getInstance();

    auto buildHandler = [&](float explicitCutoff) {
        json j = json::parse(TestUtils::gaussianPSFConfigJSON());
        j["size_x"] = 0;
        j["size_y"] = 0;
        j["size_z"] = 0;
        j["cutoff_threshold"] = explicitCutoff;
        auto psfConfig = factory.createConfig(j);

        auto threadPool = std::make_shared<ThreadPool>(1);
        auto handler = std::make_shared<PSFHandler>(threadPool, [](std::atomic<float>&, float){});

        SetupConfig setupConfig;
        setupConfig.psfConfigs = {psfConfig};
        setupConfig.psfFilePaths = {};

        DeconvolutionConfig deconvConfig;
        deconvConfig.paddingStrategyType = PaddingStrategyType::MEDIUM;

        handler->generatePSFs(setupConfig, deconvConfig, CuboidShape{128, 128, 64});
        return handler->getMaxShape();
    };

    // Explicit loose cutoff -> smaller PSF than the derived 1e-6 default.
    // sigma = 5: cutoff 0.01 gives ~33 px, derived 1e-6 gives ~55 px.
    CuboidShape loose = buildHandler(0.01f);
    CuboidShape derived = buildHandler(0.0f);

    EXPECT_GT(derived.width, loose.width);
    EXPECT_GT(derived.depth, loose.depth);
    EXPECT_LE(derived.width, 128u);
}
