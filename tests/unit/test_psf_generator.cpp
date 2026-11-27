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
#include "dolphin/psf/configs/ModalityConfig.h"
#include "dolphin/psf/generators/GibsonLanniModel.h"
#include "dolphin/psf/generators/GaussianPSFGenerator.h"
#include "dolphin/psf/generators/PSFBoxSizing.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/psf/generators/SimpsonIntegrator.h"
#include "dolphin/psf/modality/ModalityStrategy.h"
#include "dolphin/psf/modality/PSFModel.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <iostream>

using json = nlohmann::json;

namespace {
// One shared worker pool for all tests.
std::shared_ptr<ThreadPool> testPool() {
    static const std::shared_ptr<ThreadPool> pool = std::make_shared<ThreadPool>(4);
    return pool;
}

// The factory assembles a generator for the config; the pool is given via
// configure(), then the generator runs.
PSF generateTestPSF(std::shared_ptr<PSFConfig> config) {
    auto generator = PSFGeneratorFactory::getInstance().createGenerator(std::move(config));
    generator->configure(testPool(), nullptr);
    return generator->generatePSF();
}

PSF generateTestPSF(const json& j) {
    return generateTestPSF(PSFGeneratorFactory::getInstance().createConfig(j));
}

// Fast GibsonLanni config: coarser pixels, lower accuracy, bounded domain.
// All assertions are relative to the generated PSF, so this is pure effort
// reduction. size 48 keeps the PSF support (<= ~25 px) below the domain cap
// and the axial defocus at +/-12um instead of +/-24.5um.
inline json fastGlJSON() {
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["size_x"] = 48;
    j["size_y"] = 48;
    j["size_z"] = 48;
    j["pixel_size_axial_nm"] = 500.0;
    j["pixel_size_lateral_nm"] = 500.0;
    j["integration_accuracy"] = 3;
    j["oversampling_factor"] = 2;
    return j;
}

// One shared generation of the fastGlJSON() auto-size PSF. Every autosize
// test below asserts on ONE deterministic output, so they share it (and its
// generation) instead of each regenerating the identical 375x375x49 volume.
// Determinism across generations is covered by
// GibsonLanniQuadratureMatchesAdaptiveSimpson.
inline const PSF& sharedAutoSizePSF(bool normalize) {
    static const PSF normalized = [] {
        auto config = PSFGeneratorFactory::getInstance().createConfig(fastGlJSON());
        config->autoSize = true;
        config->normalizePSF = true;
        return generateTestPSF(config);
    }();
    static const PSF unnormalized = [] {
        auto config = PSFGeneratorFactory::getInstance().createConfig(fastGlJSON());
        config->autoSize = true;
        config->normalizePSF = false;
        return generateTestPSF(config);
    }();
    return normalize ? normalized : unnormalized;
}
} // namespace

class PSFGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }

    static const PSF& sharedAutoSizePSF() { return ::sharedAutoSizePSF(true); }
    static const PSF& sharedAutoSizePSFUnnormalized() { return ::sharedAutoSizePSF(false); }
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
    // Two configs that share the same ti_nm (the cache key) but differ in NA
    // must produce different PSFs: each generation gets a fresh model over
    // its own config, so no stale radial profile can leak between them.
    auto& factory = PSFGeneratorFactory::getInstance();
    auto buildConfig = [](float na) {
        auto& factory = PSFGeneratorFactory::getInstance();
        json j = fastGlJSON();
        j["size_x"] = 21;
        j["size_y"] = 21;
        j["size_z"] = 3;
        j["NA"] = na;
        return factory.createConfig(j);
    };

    PSF first = generateTestPSF(buildConfig(1.4f));
    PSF second = generateTestPSF(buildConfig(1.0f));

    EXPECT_EQ(first.getShape(), second.getShape());
    EXPECT_FALSE(first.isEqual(second, 1e-4f));
}

TEST_F(PSFGeneratorTest, GibsonLanniNoThreadPoolThrows) {
    auto& factory = PSFGeneratorFactory::getInstance();
    auto makeConfig = [&](int sizeX, int sizeY, int sizeZ) {
        json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
        j["size_x"] = sizeX;
        j["size_y"] = sizeY;
        j["size_z"] = sizeZ;
        auto config = factory.createConfig(j);
        config->cutoffThreshold = 0.05f;
        return config;
    };

    // Generation without a thread pool is rejected up front by the generator,
    // for both the auto-size and the fixed-size path.
    auto autoSizeGen = factory.createGenerator(makeConfig(0, 0, 0));
    EXPECT_THROW(autoSizeGen->generatePSF(), std::runtime_error);  // no configure() -> pool stays null
    auto fixedSizeGen = factory.createGenerator(makeConfig(33, 33, 5));
    EXPECT_THROW(fixedSizeGen->generatePSF(), std::runtime_error);
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
    PSF generator = generateTestPSF(j);

    PSF psf = generator;
    EXPECT_EQ(psf.getShape(), CuboidShape(32, 32, 16));
    EXPECT_FALSE(psf.ID.empty());
}

TEST_F(PSFGeneratorTest, GaussianPSFIsCentered) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    PSF generator = generateTestPSF(j);
    PSF psf = generator;

    float maxVal = psf.getMax();
    float centerVal = psf.getPixel(32 / 2, 32 / 2, 16 / 2);
    EXPECT_NEAR(centerVal, maxVal, maxVal * 0.01f);
}

TEST_F(PSFGeneratorTest, GaussianPSFSymmetric) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    PSF generator = generateTestPSF(j);
    PSF psf = generator;

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
    PSF generator = generateTestPSF(j);
    PSF psf = generator;

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GaussianPSFCreateFromConfig) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto config = factory.createConfig(j);
    PSF generator = generateTestPSF(config);

    PSF psf = generator;
    EXPECT_EQ(psf.getShape(), CuboidShape(32, 32, 16));
}

TEST_F(PSFGeneratorTest, GibsonLanniPSFGeneration) {
    const PSF& psf = sharedAutoSizePSF();
    CuboidShape shape = psf.getShape();
    EXPECT_GT(shape.width, 0u);
    EXPECT_GT(shape.height, 0u);
    EXPECT_GT(shape.depth, 0u);

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniPSFCentered) {
    const PSF& psf = sharedAutoSizePSF();

    CuboidShape shape = psf.getShape();
    float maxVal = psf.getMax();
    float centerVal = psf.getPixel(shape.width / 2, shape.height / 2, shape.depth / 2);
    EXPECT_NEAR(centerVal, maxVal, maxVal * 0.1f);
}

TEST_F(PSFGeneratorTest, PSFWriteAndReadTiff) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gaussianPSFConfigJSON());
    PSF generator = generateTestPSF(j);
    PSF psf = generator;

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
    json j = fastGlJSON();
    j["size_x"] = 21;
    j["size_y"] = 21;
    j["size_z"] = 5;
    PSF generator = generateTestPSF(j);

    PSF psf = generator;
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, 21u);
    EXPECT_EQ(shape.height, 21u);
    EXPECT_EQ(shape.depth, 5u);

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}


TEST_F(PSFGeneratorTest, GibsonLanniAutoSizeExactShape) {
    // Final model, fastGlJSON (NA=1.4, ni=1.515, ns=1.33, lambda=450, pitch
    // 500/500, sizeZ=48 -> zMax = 24*500 = 12000nm):
    //   tilt = max(tan(asin(1.4/1.515)), tan(asin(clamp(1.4/1.33, 0.99)))) = 7.0179
    //   resLat = 0.61*450/1.4 = 196.07nm, resAx = 450/(1.33*(1-cos(asin(0.99)))) = 393.91nm
    //   (the widefield envelopes; the point-pinhole contraction is applied by
    //   approximateResolution() itself, see ConfocalFinitePinholeKeepsWidefieldBound)
    //   halfLat = 4.94975*196.07 + 1.1*7.0179*12000 = 93607nm -> ceil(188) (above the
    //   soft cap of 150, so a warning logs; nothing is truncated)
    //   axial   = ceil(max(12000, 2*393.91)/500) = 24 (the 787.8nm floor loses to zMax)
    //   -> box 377 x 377 x 49
    const PSF& psf = sharedAutoSizePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, 377u);
    EXPECT_EQ(shape.height, 377u);
    EXPECT_EQ(shape.depth, 49u);
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoSizeGrowsWithImageDepth) {
    // A deeper imaged volume defocuses farther -> larger cone spread and
    // longer axial stack. sizeZ 48 -> 96 doubles zMax:
    //   48: zMax=12000 -> 377x377x49;  96: zMax=24000 -> 747x747x97
    // Size growth is a property of the analytic sizing model, so it is
    // asserted directly on it (a full 745x745x97 generation only to read its
    // shape cost ~30 s); the end-to-end auto-size path is covered by
    // GibsonLanniAutoSizeExactShape (48) and the AutoBoxEndToEnd tests.
    auto boxFor = [](size_t sizeZ) {
        GibsonLanniPSFConfig cfg;
        cfg.NA = 1.4f;
        cfg.ni0 = 1.515f;
        cfg.ni = 1.515f;
        cfg.ns = 1.33f;
        cfg.ng = cfg.ng0 = 1.5f;
        cfg.wavelength_nm = 450.0f;
        cfg.pixelSizeLateral_nm = 500.0f;
        cfg.pixelSizeAxial_nm = 500.0f;
        cfg.sizeZ = sizeZ;
        auto res = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, cfg.resolutionCriterion);
        return PSFBoxSizing::computeBoxHalfExtent(res, cfg);
    };

    const auto shallow = boxFor(48);
    const auto deep = boxFor(96);

    EXPECT_EQ(2 * shallow.lateral + 1, 377u);
    EXPECT_EQ(2 * shallow.depth + 1, 49u);
    EXPECT_EQ(2 * deep.lateral + 1, 747u);
    EXPECT_EQ(2 * deep.depth + 1, 97u);
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoSizeIgnoresSizeXY) {
    // sizeX/sizeY are the IMAGE size, not the PSF size: changing them must not
    // alter the autosized box shape or values in any way.
    auto& factory = PSFGeneratorFactory::getInstance();

    auto generate = [&](int sizeXY) {
        json j = fastGlJSON();
        j["size_x"] = sizeXY;
        j["size_y"] = sizeXY;
        auto config = factory.createConfig(j);
        config->autoSize = true;
        PSF generator = generateTestPSF(config);
        return generator;
    };

    PSF small = generate(16);
    PSF large = generate(600);

    EXPECT_EQ(small.getShape(), CuboidShape(377, 377, 49));
    EXPECT_EQ(large.getShape(), small.getShape());
    EXPECT_TRUE(large.isEqual(small, 1e-4f));
}

TEST_F(PSFGeneratorTest, GibsonLanniBoxHalfExtentModel) {
    // Direct unit test of the sizing function.
    // NA=1.3, ni=ns=1.33, lambda=520, pitch 100/100, sizeZ=64 (zMax=3200nm).
    // The cone uses the absolute tilt, so it grows even when RI is matched:
    //   resLat = 244nm, tilt = tan(asin(1.3/1.33)) = 4.6281
    //     halfLat = 4.94975*244 + 1.1*4.6281*3200 = 17499nm -> lat = 175
    //     (above the soft cap of 150 -> warning logs, nothing truncated)
    //   resAx = 520/(1.33*(1-cos(asin(min(1.3/1.33,0.99))))) = 544.28nm
    //     halfAx = max(3200, 2*544.28) = 3200nm -> ax = 32
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.3f;
    cfg.ni = 1.33f;
    cfg.ni0 = 1.33f;
    cfg.ns = 1.33f;
    cfg.wavelength_nm = 520.0f;
    cfg.pixelSizeLateral_nm = 100.0f;
    cfg.pixelSizeAxial_nm = 100.0f;
    cfg.sizeZ = 64;

    auto res = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, cfg.resolutionCriterion);
    auto box = PSFBoxSizing::computeBoxHalfExtent(res, cfg);
    EXPECT_EQ(box.lateral, 175u);
    EXPECT_EQ(box.depth, 32u);

    // The confocal product is narrower than the widefield one. The delivered
    // resolution already carries the point-pinhole contraction, so the
    // resulting box never grows.
    cfg.selectModality(MicroscopeModality::CONFOCAL);
    auto modalityCfg = std::dynamic_pointer_cast<ConfocalModalityConfig>(cfg.modalityConfig);
    modalityCfg->excitationWavelength_nm = 520.0f;
    modalityCfg->emissionWavelength_nm = 520.0f;
    modalityCfg->pinholeRadius_um = 0.0f;
    auto confocalRes = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, cfg.resolutionCriterion);
    auto confocalBox = PSFBoxSizing::computeBoxHalfExtent(confocalRes, cfg);
    EXPECT_LT(confocalRes.axial_nm, res.axial_nm);
    EXPECT_LE(confocalBox.lateral, box.lateral);
    EXPECT_LE(confocalBox.depth, box.depth);

    // TIR regime (NA >= ns): asin clamp keeps everything finite and positive.
    GibsonLanniPSFConfig tir = cfg;
    tir.NA = 1.4f;
    tir.ni = 1.515f;
    tir.sizeZ = 48;
    auto resTir = PSFBoxSizing::approximateResolution(tir, tir.wavelength_nm, tir.resolutionCriterion);
    auto boxTir = PSFBoxSizing::computeBoxHalfExtent(resTir, tir);
    EXPECT_GT(boxTir.lateral, 0u);
    EXPECT_GT(boxTir.depth, 0u);
    EXPECT_LT(boxTir.lateral, 10000u);
    EXPECT_LT(boxTir.depth, 10000u);
}

TEST_F(PSFGeneratorTest, GibsonLanniApproximateResolutionAxial) {
    // Axial widths in the 1-cos structure, lambda/(ns*(1-cos(alpha))) with
    // n*sin(alpha) = NA clamped at 0.99. For NA=1.4, ns=1.33 the clamp is
    // active: sin(alpha)=0.99, 1-cos=0.8589, ns*(1-cos)=1.1423:
    //   RAYLEIGH = 450/1.1423 = 393.91nm
    //   FWHM     = sqrt2*450/1.1423 = 557.08nm   (classical 0.93 form if NA<ns)
    //   SIGMA    = RAYLEIGH/sqrt2 = 278.54nm
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.4f;
    cfg.ns = 1.33f;
    cfg.wavelength_nm = 450.0f;

    auto rayleigh = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, PSFConfig::ResolutionCriterion::RAYLEIGH);
    auto fwhm = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, PSFConfig::ResolutionCriterion::FWHM);
    auto sigma = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, PSFConfig::ResolutionCriterion::SIGMA);

    EXPECT_NEAR(rayleigh.axial_nm, 393.91, 0.1);
    EXPECT_NEAR(sigma.axial_nm, 278.54, 0.1);
    EXPECT_NEAR(fwhm.axial_nm, 557.08, 0.1);
    EXPECT_GT(fwhm.axial_nm, rayleigh.axial_nm);
    EXPECT_GT(rayleigh.axial_nm, sigma.axial_nm);
    EXPECT_GT(rayleigh.lateral_nm, fwhm.lateral_nm);
    EXPECT_GT(fwhm.lateral_nm, sigma.lateral_nm);
}

TEST_F(PSFGeneratorTest, GibsonLanniAxialWidthMatchesSampleIndexForm) {
    // The axial term is the 1-cos(alpha) form with the marginal ray in the
    // sample, n*sin(alpha) = NA:
    //   RAYLEIGH: lambda / (ns*(1-cos(alpha)))
    //   FWHM:     sqrt2*lambda / (ns*(1-cos(alpha)))
    // and for NA << ns both reduce to the familiar n*lambda/NA^2 order.
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.2f;
    cfg.ns = 1.33f;
    cfg.wavelength_nm = 520.0f;

    const double lambda = 520.0, NA = 1.2, ns = 1.33;
    const double alpha = std::asin(NA / ns);
    const double rayleigh = PSFBoxSizing::axialNWidth_nm(lambda, NA, ns);
    const double fwhm = PSFBoxSizing::axialFwhm_nm(lambda, NA, ns);

    EXPECT_NEAR(rayleigh, lambda / (ns * (1.0 - std::cos(alpha))), 1e-9);
    EXPECT_NEAR(fwhm, std::sqrt(2.0) * lambda / (ns * (1.0 - std::cos(alpha))), 1e-9);
    // The 1-cos form is genuinely more accurate than the small-NA closed form
    // n*lambda/NA^2: at NA/ns ~ 0.9 it is 1.43x larger, not a rounding difference.
    EXPECT_GT(rayleigh, ns * lambda / (NA * NA) * 1.4);

    // Above the sample index (oil into water) the clamp keeps the term finite
    // and positive instead of sqrt of a negative number.
    EXPECT_GT(PSFBoxSizing::axialNWidth_nm(lambda, 1.4, ns), 0.0);
    EXPECT_GT(PSFBoxSizing::axialFwhm_nm(lambda, 1.4, ns), 0.0);
}

namespace {
// Optics shared by the confocal-resolution tests (OilConfocal geometry).
GibsonLanniPSFConfig confocalTestConfig() {
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.4f;
    cfg.ni = 1.515f;
    cfg.ni0 = 1.515f;
    cfg.ns = 1.33f;
    cfg.ng = 1.5f;
    cfg.ng0 = 1.5f;
    cfg.wavelength_nm = 520.0f;
    cfg.pixelSizeLateral_nm = 100.0f;
    cfg.pixelSizeAxial_nm = 100.0f;
    cfg.sizeZ = 16;
    return cfg;
}

// Switch a config to a confocal modality with the given wavelengths (nm).
void makeConfocal(GibsonLanniPSFConfig& cfg, float excitation_nm, float emission_nm) {
    cfg.selectModality(MicroscopeModality::CONFOCAL);
    auto modalityCfg = std::dynamic_pointer_cast<ConfocalModalityConfig>(cfg.modalityConfig);
    modalityCfg->excitationWavelength_nm = excitation_nm;
    modalityCfg->emissionWavelength_nm = emission_nm;
    modalityCfg->pinholeRadius_um = 0.0f;
}
} // namespace

TEST_F(PSFGeneratorTest, SamplingLimitsAreOtfcutoffHalves) {
    // The sampling limits are derived from the incoherent OTF bandwidths
    // (lateral 2*NA/lambda, axial n*(1-cos(alpha))/lambda per integral; the
    // confocal product adds the excitation and detection bandwidths). The
    // Abbe (finest period) is 1/cutoff, the Nyquist pitch is half of it.
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.4f;
    cfg.ns = 1.33f;
    cfg.wavelength_nm = 520.0f;

    const double NA = 1.4, ns = 1.33, lambda = 520.0;
    const double s = std::clamp(NA / ns, 0.0, 0.99);  // the implementation's clamp
    const double omc = 1.0 - std::sqrt(1.0 - s * s);

    // Widefield: one factor on each axis.
    const auto wf = PSFBoxSizing::samplingLimits(cfg, lambda);
    EXPECT_NEAR(wf.abbeLateral_nm, lambda / (2.0 * NA), 0.01);
    EXPECT_NEAR(wf.nyquistLateral_nm, lambda / (4.0 * NA), 0.01);
    EXPECT_NEAR(wf.abbeAxial_nm, lambda / (ns * omc), 0.5);
    EXPECT_NEAR(wf.nyquistAxial_nm, lambda / (2.0 * ns * omc), 0.5);
    // Nyquist is exactly half the Abbe finest period.
    EXPECT_DOUBLE_EQ(wf.nyquistLateral_nm, 0.5 * wf.abbeLateral_nm);
    EXPECT_DOUBLE_EQ(wf.nyquistAxial_nm, 0.5 * wf.abbeAxial_nm);

    // Confocal point pinhole: the bandwidths add. 488/520 ->
    //   lateral Nyquist = 1 / (4*NA*(1/488 + 1/520)) = 44.955nm
    //   axial   Nyquist = 1 / (2*n*(1-cos)*(1/488 + 1/520)) = 110.18nm
    // (both exactly the Delta_x / Delta_z of the confocal sampling-limit
    // equations) and the Abbe is exactly double the Nyquist on each axis -
    // halved vs the widefield Abbe for equal wavelengths, the OTF form of
    // "confocal doubles the resolution".
    GibsonLanniPSFConfig cf = cfg;
    makeConfocal(cf, 488.0f, 520.0f);

    const auto c = PSFBoxSizing::samplingLimits(cf, lambda);
    const double sumInv = 1.0 / 488.0 + 1.0 / 520.0;
    EXPECT_NEAR(c.nyquistLateral_nm, 1.0 / (4.0 * NA * sumInv), 0.01);
    EXPECT_NEAR(c.nyquistAxial_nm, 1.0 / (2.0 * ns * omc * sumInv), 0.5);
    EXPECT_DOUBLE_EQ(c.abbeLateral_nm, 2.0 * c.nyquistLateral_nm);
    EXPECT_DOUBLE_EQ(c.abbeAxial_nm, 2.0 * c.nyquistAxial_nm);

    // Equal-wavelength confocal: Abbe exactly half the widefield Abbe.
    makeConfocal(cf, 520.0f, 520.0f);
    const auto eq = PSFBoxSizing::samplingLimits(cf, lambda);
    EXPECT_NEAR(eq.abbeLateral_nm, wf.abbeLateral_nm / 2.0, 0.01);
    EXPECT_NEAR(eq.abbeAxial_nm, wf.abbeAxial_nm / 2.0, 0.5);
}

TEST_F(PSFGeneratorTest, ConfocalResolutionContractsPointPinhole) {
    // The confocal effective PSF is the product of the excitation and
    // detection PSFs. For equal wavelengths its width contracts by exactly
    // 1/sqrt(2) relative to the widefield single-integral resolution (the
    // literature confocal values 0.4*lambda/NA lateral and 1.4*n*lambda/NA^2
    // axial are these contracted widefield constants), so the box-sizing
    // resolution must shrink accordingly.
    //
    // The contraction is a pure wavelength-ratio effect, so applying it to the
    // widefield return must reproduce the contracted width exactly.
    GibsonLanniPSFConfig cfg = confocalTestConfig();
    const double lambda = 520.0;

    const auto widefield = PSFBoxSizing::approximateResolution(cfg, lambda, cfg.resolutionCriterion);
    const auto contracted = PSFBoxSizing::confocalResolution(widefield, 1.0 / std::sqrt(2.0));

    EXPECT_NEAR(contracted.lateral_nm, widefield.lateral_nm / std::sqrt(2.0), 0.1);
    EXPECT_NEAR(contracted.axial_nm, widefield.axial_nm / std::sqrt(2.0), 0.1);

    // The contraction holds for every criterion (FWHM: 0.51 -> 0.36 lambda/NA).
    const auto widefieldFwhm = PSFBoxSizing::approximateResolution(cfg, lambda, PSFConfig::ResolutionCriterion::FWHM);
    const auto confocalFwhm = PSFBoxSizing::confocalResolution(widefieldFwhm, 1.0 / std::sqrt(2.0));
    EXPECT_NEAR(confocalFwhm.lateral_nm, 0.51 * (lambda / std::sqrt(2.0)) / 1.4, 0.2);
    EXPECT_LT(confocalFwhm.lateral_nm, widefieldFwhm.lateral_nm);
}

TEST_F(PSFGeneratorTest, ConfocalResolutionCombinesStokesShiftedWavelengths) {
    // With a Stokes shift the confocal product resolves at the combination of
    // BOTH integrals' resolutions:
    //   res_eff = res(exc) * res(em) / sqrt(res(exc)^2 + res(em)^2)
    //           = 0.61*lambda_eff/NA  with  lambda_eff = exc*em/sqrt(exc^2+em^2)
    // 488/520 -> lambda_eff = 355.83nm, point pinhole (radius 0):
    //   lateral RAYLEIGH = 0.61*355.83/1.4 = 155.04nm
    //   axial   RAYLEIGH = 355.83/(1.33*(1-cos(asin(1.4/1.33 clamp)))) = 311.49nm
    GibsonLanniPSFConfig cfg = confocalTestConfig();
    makeConfocal(cfg, 488.0f, 520.0f);

    const auto res = PSFBoxSizing::approximateResolution(cfg, 520.0, cfg.resolutionCriterion);
    EXPECT_NEAR(res.lateral_nm, 155.04, 0.2);
    EXPECT_NEAR(res.axial_nm, 311.49, 0.2);

    // Degenerate fallback (excitation <= 0 -> emission wavelength): the
    // product degenerates to two identical integrals -> the 1/sqrt(2)
    // contraction of the detection resolution.
    makeConfocal(cfg, 0.0f, 520.0f);
    const auto degenerate = PSFBoxSizing::approximateResolution(cfg, 520.0, cfg.resolutionCriterion);
    GibsonLanniPSFConfig widefieldCfg = confocalTestConfig();
    const auto widefield = PSFBoxSizing::approximateResolution(widefieldCfg, 520.0, widefieldCfg.resolutionCriterion);
    EXPECT_NEAR(degenerate.lateral_nm, widefield.lateral_nm / std::sqrt(2.0), 0.1);
}

TEST_F(PSFGeneratorTest, ConfocalFinitePinholeKeepsWidefieldBound) {
    // The delivered resolution is the CONFOCAL width for a closed (point)
    // pinhole, and the WIDEFIELD envelope for a finite one: a finite pinhole
    // re-admits the detection blur, so the box must keep bounding the wider
    // defocused off-axis support. Hence:
    //   res(point pinhole) < res(finite pinhole) == res(widefield)
    // and the finite box is at least as large as the point-pinhole box.
    GibsonLanniPSFConfig point = confocalTestConfig();
    makeConfocal(point, 520.0f, 520.0f);

    GibsonLanniPSFConfig finite = confocalTestConfig();
    makeConfocal(finite, 520.0f, 520.0f);
    std::dynamic_pointer_cast<ConfocalModalityConfig>(finite.modalityConfig)->pinholeRadius_um = 0.3f;

    GibsonLanniPSFConfig widefield = confocalTestConfig();

    const auto pointRes = PSFBoxSizing::approximateResolution(point, 520.0, point.resolutionCriterion);
    const auto finiteRes = PSFBoxSizing::approximateResolution(finite, 520.0, finite.resolutionCriterion);
    const auto wideRes = PSFBoxSizing::approximateResolution(widefield, 520.0, widefield.resolutionCriterion);

    // Point pinhole contracts toward widefield by exactly 1/sqrt2.
    EXPECT_NEAR(pointRes.lateral_nm, wideRes.lateral_nm / std::sqrt(2.0), 0.1);
    EXPECT_NEAR(pointRes.axial_nm, wideRes.axial_nm / std::sqrt(2.0), 0.1);
    // Finite pinhole keeps the widefield envelope (no contraction).
    EXPECT_NEAR(finiteRes.lateral_nm, wideRes.lateral_nm, 1e-9);
    EXPECT_NEAR(finiteRes.axial_nm, wideRes.axial_nm, 1e-9);

    const auto pointBox = PSFBoxSizing::computeBoxHalfExtent(pointRes, point);
    const auto finiteBox = PSFBoxSizing::computeBoxHalfExtent(finiteRes, finite);
    // The finite-pinhole box bounds the point-pinhole box (and both bound the
    // defocused support).
    EXPECT_GE(finiteBox.lateral, pointBox.lateral);
    EXPECT_GE(finiteBox.depth, pointBox.depth);
    EXPECT_GT(finiteBox.depth - pointBox.depth, 0u);
}

TEST_F(PSFGeneratorTest, ConfocalResolutionShrinksAutoSizeBox) {
    // The delivered resolution is already contracted for a point pinhole, so
    // the auto-sized box shrinks accordingly (ring term and axial floor; the
    // geometric defocus cone dominates laterally and is unchanged).
    // OilConfocal depth-16 geometry: widefield 147x21 -> confocal 141x17
    //   resLat 226.57 -> 160.21nm (x 1/sqrt2), resAx 455.19 -> 321.87nm
    //   halfLat = 4.94975*160.21 + 1.1*7.0179*800 = 6969nm -> 70 (box 141)
    //   halfAx  = max(800, 2*321.87) = 800nm    -> 8 (box 17)
    GibsonLanniPSFConfig cfg = confocalTestConfig();

    const auto widefieldRes = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, cfg.resolutionCriterion);
    const auto widefieldBox = PSFBoxSizing::computeBoxHalfExtent(widefieldRes, cfg);
    EXPECT_EQ(2 * widefieldBox.lateral + 1, 147u);
    EXPECT_EQ(2 * widefieldBox.depth + 1, 21u);

    makeConfocal(cfg, 520.0f, 520.0f);  // point pinhole
    const auto confocalRes = PSFBoxSizing::approximateResolution(cfg, 520.0, cfg.resolutionCriterion);
    const auto confocalBox = PSFBoxSizing::computeBoxHalfExtent(confocalRes, cfg);

    EXPECT_EQ(2 * confocalBox.lateral + 1, 141u);
    EXPECT_EQ(2 * confocalBox.depth + 1, 17u);
    EXPECT_LE(confocalBox.lateral, widefieldBox.lateral);
    EXPECT_LT(confocalBox.depth, widefieldBox.depth);
}

// Result box sizes for realistic microscope configurations. Values are the
// exact output of computeBoxHalfExtent for each config (RAYLEIGH criterion
// unless noted); run this test to see the full size table in the log.
//
// Formula (PSFBoxSizing):
//   tilt     = max(tan(asin(clamp(NA/ni))), tan(asin(clamp(NA/ns)))) (clamp 0.99)
//   zMax     = max(sizeZ/2, sizeZ-1-sizeZ/2) * dz
//   resLat   = 0.61*lambda/NA (0.51 / 0.408 by criterion)  - widefield only
    //   resAx    = lambda/(ns*(1-cos(alpha))), n*sin(alpha) = NA  - widefield only
    //   halfLat  = 4.9497*resLat + 1.1*tilt*zMax          (>= 4.9497*resLat)
    //   halfAx   = max(zMax, 2*resAx)                     (voxel-ceil)
    //   box      = (2*ceil(halfLat/pxLat)+1) x (2*ceil(halfAx/pxAx)+1)
    // 4.9497 = RING_FACTOR * sqrt2: the ring count doubles in quadrature along
    // the diagonal, which is the direction that decides the square box side.
    // The axial width is the 1-cos(alpha) form with the marginal ray in the
    // sample (sin(alpha) = NA/ns, clamped at 0.99), NOT the classic
    // 2*n*lambda/NA^2 depth-of-focus.
// Nothing is truncated; configs whose envelope exceeds the soft cap
// (SOFT_MAX_LATERAL_HALF_VOXELS) log a warning.
TEST_F(PSFGeneratorTest, GibsonLanniAutoBoxSizeCatalog) {
    struct AutoBoxCase {
        const char* name;
        float NA, ni, ns, lambda_nm, pxLat_nm, pxAx_nm;
        size_t sizeZ;
        int criterion;
        size_t expectW, expectD;  // full box width (X=Y) and depth (Z)
    };
    using C = PSFConfig::ResolutionCriterion;

    //          name                     NA    ni      ns     lam  pxLat pxAx depth criterion     W    D
    const std::vector<AutoBoxCase> cases = {
        {"WidefieldDry_NA055",        0.55f, 1.0f,   1.338f, 520, 100, 500, 100, C::RAYLEIGH, 421, 101},
        {"DefaultGibsonLanniConfig",  0.55f, 1.43f,  1.338f, 520, 100, 500, 100, C::RAYLEIGH, 307, 101},
        {"OilConfocal_NA14",          1.40f, 1.515f, 1.33f,  520, 100, 100,  64, C::RAYLEIGH, 519, 65},
        {"MatchedRI_coneStillGrows",  1.30f, 1.33f,  1.33f,  520, 100, 100,  64, C::RAYLEIGH, 351, 65},
        {"TIRclamp_NAaboveNs",        1.40f, 1.515f, 1.33f,  450, 100, 100,  64, C::RAYLEIGH, 515, 65},
        {"OilConfocal_FWHM",          1.40f, 1.515f, 1.33f,  520, 100, 100,  64, C::FWHM,     515, 65},
        {"OilConfocal_Sigma",         1.40f, 1.515f, 1.33f,  520, 100, 100,  64, C::SIGMA,    511, 65},
        {"OilConfocal_coarsePitch",   1.40f, 1.515f, 1.33f,  520, 200, 200,  64, C::RAYLEIGH, 507, 65},
        {"OilConfocal_depth128",      1.40f, 1.515f, 1.33f,  520, 100, 100, 128, C::RAYLEIGH, 1013, 129},
        {"OilConfocal_depth16",       1.40f, 1.515f, 1.33f,  520, 100, 100,  16, C::RAYLEIGH, 147, 21},
    };

    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);

        GibsonLanniPSFConfig cfg;
        cfg.NA = c.NA;
        cfg.ni = c.ni;
        cfg.ni0 = c.ni;
        cfg.ns = c.ns;
        cfg.ng = 1.5f;
        cfg.ng0 = 1.5f;
        cfg.wavelength_nm = c.lambda_nm;
        cfg.pixelSizeLateral_nm = c.pxLat_nm;
        cfg.pixelSizeAxial_nm = c.pxAx_nm;
        cfg.sizeZ = c.sizeZ;

        auto res = PSFBoxSizing::approximateResolution(cfg, cfg.wavelength_nm, c.criterion);
        auto box = PSFBoxSizing::computeBoxHalfExtent(res, cfg);

        std::cout << "[AutoBox] " << c.name
                  << "  NA=" << c.NA << " ni=" << c.ni << " ns=" << c.ns
                  << " lam=" << c.lambda_nm << "nm"
                  << " px=" << c.pxLat_nm << "/" << c.pxAx_nm << "nm"
                  << " depth=" << c.sizeZ
                  << "  ->  box " << (2 * box.lateral + 1)
                  << " x " << (2 * box.lateral + 1)
                  << " x " << (2 * box.depth + 1) << " voxels"
                  << "  (resLat=" << res.lateral_nm << "nm resAx=" << res.axial_nm
                  << "nm zMax=" << (c.sizeZ / 2 * static_cast<size_t>(c.pxAx_nm)) << "nm)"
                  << std::endl;

        EXPECT_EQ(2 * box.lateral + 1, c.expectW);
        EXPECT_EQ(2 * box.depth + 1, c.expectD);
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoBoxEndToEndOilConfocal) {
    // Full generation must match the analytic catalog row "OilConfocal_NA14":
    // NA=1.4, ni=1.515, ns=1.33, lambda=520, pitch 100/100, depth 64 ->
    // zMax=3200nm, tilt=7.0179, halfLat=4.9497*226.57+1.1*7.0179*3200=25825nm
    // -> ceil(259) (above the soft cap of 150 -> warning logs, nothing
    // truncated); axial=ceil(max(3200,2*455.19)/100)=32
    // -> box 519 x 519 x 65.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    j["size_x"] = 64;
    j["size_y"] = 64;
    j["size_z"] = 64;
    j["wavelength_nm"] = 520.0;
    j["pixel_size_axial_nm"] = 100.0;
    j["pixel_size_lateral_nm"] = 100.0;
    auto config = factory.createConfig(j);
    config->autoSize = true;
    PSF generator = generateTestPSF(config);

    PSF psf = generator;
    EXPECT_EQ(psf.getShape(), CuboidShape(519, 519, 65));
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoBoxEndToEndDefaultConfig) {
    // The DefaultGibsonLanniConfig (configs/default_psf_gibsonlanni.json):
    // NA=0.55, ni=1.43, ns=1.338, lambda=520, pitch 100/500nm, image
    // 720x672x100. sizeX/sizeY must not matter; sizeZ=100 drives the defocus.
    //   resLat = 0.61*520/0.55   = 576.73nm
    //   resAx  = lambda/(ns*(1-cos(alpha))), n*sin(alpha) = NA = 4396.76nm
    //   zMax   = 50 * 500nm      = 25000nm (mid-stack focus)
    //   tilt   = max(tan(asin(0.55/1.43)), tan(asin(0.55/1.338))) = 0.45092
    //   halfLat = 4.9497*576.73 + 1.1*0.45092*25000 = 15255nm -> lat = 153 (< cap)
    //   axial   = ceil(max(25000, 2*4600.07)/500) = 50
    //   -> box 307 x 307 x 101
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(R"({
        "model_name": "GibsonLanni",
        "id": "DefaultGibsonLanniConfig",
        "size_x": 720,
        "size_y": 672,
        "size_z": 100,
        "NA": 0.55,
        "wavelength_nm": 520.0,
        "integration_accuracy": 5,
        "working_distance_design_nm": 0,
        "working_distance_experimental_nm": 0,
        "immersion_ri_design": 1.43,
        "immersion_ri_experimental": 1.43,
        "coverslip_thickness_design_nm": 170000.0,
        "coverslip_thickness_experimental_nm": 170000.0,
        "coverslip_ri_design": 1.5,
        "coverslip_ri_experimental": 1.5,
        "sample_ri": 1.338,
        "particle_axial_position_nm": 0.0,
        "pixel_size_axial_nm": 500.0,
        "pixel_size_lateral_nm": 100.0
    })");
    auto config = factory.createConfig(j);
    config->autoSize = true;  // PSFHandler forces this in production
    PSF generator = generateTestPSF(config);

    PSF psf = generator;
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, 307u);
    EXPECT_EQ(shape.height, 307u);
    EXPECT_EQ(shape.depth, 101u);

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniAnalyticBoxIsSymmetric) {
    const PSF& psf = sharedAutoSizePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, shape.height);
    EXPECT_EQ(shape.depth % 2, 1u);
}

TEST_F(PSFGeneratorTest, GibsonLanniBoxAllValuesComputed) {
    // Every voxel inside the box must be a real integrated value (no zero
    // placeholder from per-slice disk shrinking). With a unit-flux normalised
    // PSF the far in-box corner is small but strictly positive.
    // Unnormalised: normalisation preserves the zero/non-zero pattern but the
    // dedicated unnormalised PSF keeps this test independent of it.
    const PSF& psf = sharedAutoSizePSFUnnormalized();
    CuboidShape shape = psf.getShape();

    // The lateral edge midpoint sits at exactly the box half-width (radius =
    // half), the outermost computed sample of each slice. It must hold a real
    // integrated value, proving slices are integrated to the global half-width
    // rather than shrunk to a per-slice disk.
    // (Box corners lie at radius sqrt(2)*half, beyond the radial support, and
    // are legitimately zero.)
    float lateralEdge = psf.getPixel(0, shape.height / 2, shape.depth / 2);
    EXPECT_GT(lateralEdge, 0.0f);
}


TEST_F(PSFGeneratorTest, GibsonLanniPeakNearCenter) {
    const PSF& psf = sharedAutoSizePSF();

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

TEST_F(PSFGeneratorTest, GibsonLanniUnitFluxNormalization) {
    const PSF& psf = sharedAutoSizePSF();

    double sum = 0.0;
    for (auto it = psf.cbegin(); it != psf.cend(); ++it)
        sum += *it;

    EXPECT_NEAR(sum, 1.0, 1e-4);
}

TEST_F(PSFGeneratorTest, GibsonLanniResolutionCriterionFromJSON) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["resolution_criterion"] = "fwhm";
    auto config = factory.createConfig(j);
    EXPECT_EQ(config->resolutionCriterion, PSFConfig::ResolutionCriterion::FWHM);
}

TEST_F(PSFGeneratorTest, GibsonLanniCenteredPeak) {
    // The symmetric analytic box anchors the PSF on the reference slice; the
    // on-axis peak must stay within a couple of slices of the center.
    const PSF& psf = sharedAutoSizePSF();

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
    j["size_x"] = 17;
    j["size_y"] = 17;
    j["size_z"] = 3;
    j["integration_accuracy"] = 2;
    j["pixel_size_lateral_nm"] = 500.0;
    j["oversampling_factor"] = 2;
    auto config = factory.createConfig(j);

    // The fixed-node quadrature is the only radial integration path; assert
    // it is deterministic and numerically clean across generations.
    PSF quadraturePSF = generateTestPSF(config);
    PSF regeneratedPSF = generateTestPSF(config);

    EXPECT_EQ(quadraturePSF.getShape(), regeneratedPSF.getShape());
    EXPECT_TRUE(quadraturePSF.isEqual(regeneratedPSF, 1e-6f));
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

        auto deconvConfig = DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm("RichardsonLucy");
        deconvConfig->paddingStrategyType = PaddingStrategyType::MEDIUM;

        handler->generatePSFs(setupConfig, *deconvConfig, CuboidShape{128, 128, 64});
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

// ---------------------------------------------------------------------------
// Confocal Gibson-Lanni (product of excitation and emission integrals)
// ---------------------------------------------------------------------------

namespace {

// Auto-size confocal-capable config with pitch fine enough to resolve FWHM.
inline json confocalGlJSON(bool confocal, double pinholeRadius_um = 0.28) {
    json j = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    j["NA"] = 0.7;
    j["immersion_ri_design"] = 1.515;
    j["immersion_ri_experimental"] = 1.515;
    j["sample_ri"] = 1.338;
    j["pixel_size_axial_nm"] = 100.0;
    j["pixel_size_lateral_nm"] = 100.0;
    j["integration_accuracy"] = 4;
    j["wavelength_nm"] = 520.0;
    j["microscope_modality"] = confocal ? "confocal" : "widefield";
    if (confocal) {
        // The confocal wavelengths and pinhole belong to the modality config;
        // widefield configs do not know these keys.
        j["emission_wavelength_nm"] = j["wavelength_nm"];
        j.erase("wavelength_nm");
        j["excitation_wavelength_nm"] = 488.0;
        j["pinhole_radius_um"] = pinholeRadius_um;
    }
    return j;
}

double fwhmAlongAxis(const PSF& psf, int axis, double pitch_nm) {
    CuboidShape shape = psf.getShape();
    const size_t dims[3] = {shape.width, shape.height, shape.depth};

    // Global peak voxel.
    float peak = 0.0f;
    size_t at[3] = {0, 0, 0};
    for (size_t z = 0; z < shape.depth; z++)
        for (size_t y = 0; y < shape.height; y++)
            for (size_t x = 0; x < shape.width; x++) {
                float v = psf.getPixel(x, y, z);
                if (v > peak) { peak = v; at[0] = x; at[1] = y; at[2] = z; }
            }

    // Profile through the peak along the requested axis.
    std::vector<double> profile(dims[axis]);
    for (size_t i = 0; i < dims[axis]; i++) {
        size_t p[3] = {at[0], at[1], at[2]};
        p[axis] = i;
        profile[i] = psf.getPixel(p[0], p[1], p[2]);
    }

    const double half = profile[at[axis]] / 2.0;
    auto crossing = [&](long start, long step) -> double {
        for (long i = start; i >= 0 && i < static_cast<long>(dims[axis]) - 1; i += step) {
            if (profile[i] < half) {
                // a = last value above half, b = first below; walk from a
                // toward b by the fractional half-max position (step-signed).
                const double a = profile[i - step];
                const double b = profile[i];
                const double t = (a - half) / (a - b);
                return static_cast<double>(i - step) + static_cast<double>(step) * t;
            }
        }
        return step > 0 ? static_cast<double>(dims[axis] - 1) : 0.0;
    };

    const double right = crossing(static_cast<long>(at[axis]), 1);
    const double left = crossing(static_cast<long>(at[axis]), -1);
    return (right - left) * pitch_nm;
}

struct ConfocalMetrics {
    double lateralFWHM_nm;
    double axialFWHM_nm;
    double central3Fraction;  // energy in the 3 slices around the focal plane
    double peak;
};

ConfocalMetrics measureMetrics(const PSF& psf, double pitchLateral_nm, double pitchAxial_nm) {
    ConfocalMetrics m;
    m.lateralFWHM_nm = fwhmAlongAxis(psf, 0, pitchLateral_nm);
    m.axialFWHM_nm = fwhmAlongAxis(psf, 2, pitchAxial_nm);
    m.peak = psf.getMax();

    double total = 0.0;
    for (auto it = psf.cbegin(); it != psf.cend(); ++it)
        total += *it;

    CuboidShape shape = psf.getShape();
    // Focal plane proxy: the peak slice (tests allow +-2 slices off center).
    size_t peakZ = 0;
    double peakSliceSum = -1.0;
    for (size_t z = 0; z < shape.depth; z++) {
        double s = 0.0;
        for (size_t y = 0; y < shape.height; y++)
            for (size_t x = 0; x < shape.width; x++)
                s += psf.getPixel(x, y, z);
        if (s > peakSliceSum) { peakSliceSum = s; peakZ = z; }
    }
    double central3 = 0.0;
    if (peakZ >= 1) for (size_t y = 0; y < shape.height; y++)
        for (size_t x = 0; x < shape.width; x++)
            central3 += psf.getPixel(x, y, peakZ - 1);
    for (size_t y = 0; y < shape.height; y++)
        for (size_t x = 0; x < shape.width; x++)
            central3 += psf.getPixel(x, y, peakZ);
    if (peakZ + 1 < shape.depth) for (size_t y = 0; y < shape.height; y++)
        for (size_t x = 0; x < shape.width; x++)
            central3 += psf.getPixel(x, y, peakZ + 1);

    m.central3Fraction = total > 0.0 ? central3 / total : 0.0;
    return m;
}

} // namespace

TEST_F(PSFGeneratorTest, ModalityStrategySelection) {
    // The generator delegates PSF assembly to the modality strategy selected
    // from the config; each modality must resolve to its singleton.
    GibsonLanniPSFConfig widefield;
    EXPECT_EQ(&ModalityStrategy::forConfig(widefield), &WidefieldModality::instance());

    GibsonLanniPSFConfig confocal;
    confocal.selectModality(MicroscopeModality::CONFOCAL);
    EXPECT_EQ(&ModalityStrategy::forConfig(confocal), &ConfocalModality::instance());

    // Confocal identity: the typed modality config carries the parameters,
    // excitation wavelength resolution falls back to the emission wavelength.
    ConfocalModalityConfig* modalityCfg = dynamic_cast<ConfocalModalityConfig*>(confocal.modalityConfig.get());
    ASSERT_NE(modalityCfg, nullptr);
    modalityCfg->excitationWavelength_nm = 488.0f;
    modalityCfg->emissionWavelength_nm = 520.0f;
    EXPECT_DOUBLE_EQ(ConfocalModality::excitationLambda(confocal), 488.0);
    modalityCfg->excitationWavelength_nm = 0.0f;
    EXPECT_DOUBLE_EQ(ConfocalModality::excitationLambda(confocal), 520.0);
}

namespace {

// Non-radially-symmetric, wavelength-less stub model: every slice is an
// explicit FullSlice with a bright column at x=1 and zeros elsewhere. Proves
// the modality layer composes full slabs without radial assumptions.
class AsymmetricStubModel : public PSFModel {
public:
    void init(const PSFConfig&, size_t, size_t, std::optional<double>) const override {}

    void validateConfig(const PSFConfig&) const override {}

    ModelSlice intensitySlice(const PSFConfig&, std::optional<double>, size_t,
                              size_t maxLateralPixels, size_t forcedCutoff) const override {
        const size_t cutoff = forcedCutoff > 0 ? forcedCutoff : maxLateralPixels;
        const size_t width = 2 * cutoff + 1;
        FullSlice slice;
        slice.width = width;
        slice.height = width;
        slice.data.assign(width * width, 0.0f);
        for (size_t y = 0; y < width; ++y)
            slice.data[y * width + 1] = 1.0f;
        return slice;
    }

    ModelBox autoBox(const PSFConfig&, std::optional<double>) const override {
        throw std::logic_error("AsymmetricStubModel has no auto-size box");
    }

    std::shared_ptr<const PSFConfig> sliceConfig(const PSFConfig& config, long) const override {
        // No defocus parameterization: every slice uses the same config.
        return std::make_shared<const PSFConfig>(config);
    }

    std::optional<double> detectionLambda(const PSFConfig&) const override {
        return std::nullopt;   // wavelength-less model
    }
};

} // namespace

TEST_F(PSFGeneratorTest, WidefieldModalityComposesFullSlabWithoutRadialAssumptions) {
    // A non-symmetric, wavelength-less model must flow through the widefield
    // modality unchanged: the bright column stays off-center (asymmetry
    // preserved).
    GibsonLanniPSFConfig cfg;
    cfg.sizeX = 5;
    cfg.sizeY = 5;
    cfg.sizeZ = 3;
    cfg.normalizePSF = false;

    AsymmetricStubModel model;
    ThreadPool threadPool(2);
    ProgressTracking progress;

    PSF psf = WidefieldModality::instance().generate(model, cfg, threadPool, progress);

    CuboidShape shape = psf.getShape();
    EXPECT_EQ(shape, CuboidShape(5, 5, 3));
    for (size_t z = 0; z < shape.depth; ++z) {
        for (size_t y = 0; y < shape.height; ++y) {
            EXPECT_GT(psf.getPixel(1, y, z), 0.0f);   // bright column
            EXPECT_EQ(psf.getPixel(0, y, z), 0.0f);   // asymmetric: left dark
            EXPECT_EQ(psf.getPixel(3, y, z), 0.0f);   // asymmetric: right dark
        }
    }
}

TEST_F(PSFGeneratorTest, ConfocalModalityRejectsWavelengthLessModel) {
    // Confocal image formation needs two wavelengths; pairing it with a
    // wavelength-less model must fail with a clear error instead of silently
    // composing garbage.
    GibsonLanniPSFConfig cfg;
    cfg.sizeX = 5;
    cfg.sizeY = 5;
    cfg.sizeZ = 3;
    cfg.normalizePSF = false;
    cfg.selectModality(MicroscopeModality::CONFOCAL);

    AsymmetricStubModel model;
    ThreadPool threadPool(2);
    ProgressTracking progress;

    EXPECT_THROW(ConfocalModality::instance().generate(model, cfg, threadPool, progress),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// validateConfig: modality strategies, models, and the generation entry points
// ---------------------------------------------------------------------------

TEST_F(PSFGeneratorTest, ModalityValidateConfigAcceptsDefaults) {
    // Default widefield and confocal configs are valid.
    GibsonLanniPSFConfig widefield;
    EXPECT_NO_THROW(WidefieldModality::instance().validateConfig(widefield));

    GibsonLanniPSFConfig confocal;
    confocal.selectModality(MicroscopeModality::CONFOCAL);
    EXPECT_NO_THROW(ConfocalModality::instance().validateConfig(confocal));
}

TEST_F(PSFGeneratorTest, ConfocalValidateConfigRejectsMissingModalityConfig) {
    // A widefield config carries no confocal modality config; pointing the
    // confocal strategy at it must fail loudly.
    GibsonLanniPSFConfig cfg;   // widefield
    EXPECT_THROW(ConfocalModality::instance().validateConfig(cfg), std::runtime_error);
}

TEST_F(PSFGeneratorTest, ConfocalValidateConfigRejectsNonsenseParameters) {
    GibsonLanniPSFConfig cfg;
    cfg.selectModality(MicroscopeModality::CONFOCAL);
    auto* modalityCfg = dynamic_cast<ConfocalModalityConfig*>(cfg.modalityConfig.get());
    ASSERT_NE(modalityCfg, nullptr);

    modalityCfg->emissionWavelength_nm = 0.0f;
    EXPECT_THROW(ConfocalModality::instance().validateConfig(cfg), std::runtime_error);

    modalityCfg->emissionWavelength_nm = 520.0f;
    modalityCfg->pinholeRadius_um = -1.0f;
    EXPECT_THROW(ConfocalModality::instance().validateConfig(cfg), std::runtime_error);

    // Zero pinhole is the defined point-pinhole limit: valid.
    modalityCfg->pinholeRadius_um = 0.0f;
    modalityCfg->excitationWavelength_nm = 488.0f;
    EXPECT_NO_THROW(ConfocalModality::instance().validateConfig(cfg));
}

TEST_F(PSFGeneratorTest, GibsonLanniModelValidateConfig) {
    GibsonLanniPSFConfig cfg;
    GibsonLanniModel model;

    EXPECT_NO_THROW(model.validateConfig(cfg));

    cfg.NA = 0.0f;
    EXPECT_THROW(model.validateConfig(cfg), std::runtime_error);

    cfg.NA = 1.0f;
    cfg.pixelSizeLateral_nm = 0.0f;
    EXPECT_THROW(model.validateConfig(cfg), std::runtime_error);

    cfg.pixelSizeLateral_nm = 100.0f;
    cfg.ti_nm = -1.0f;
    EXPECT_THROW(model.validateConfig(cfg), std::runtime_error);

    cfg.ti_nm = 150000.0f;
    cfg.ns = 0.0f;
    EXPECT_THROW(model.validateConfig(cfg), std::runtime_error);
}

TEST_F(PSFGeneratorTest, GenerateRunsValidateConfig) {
    // generate() must reject an invalid config before any computation.
    GibsonLanniPSFConfig cfg;
    cfg.selectModality(MicroscopeModality::CONFOCAL);
    auto* modalityCfg = dynamic_cast<ConfocalModalityConfig*>(cfg.modalityConfig.get());
    ASSERT_NE(modalityCfg, nullptr);
    modalityCfg->emissionWavelength_nm = -1.0f;

    GibsonLanniModel model;
    ThreadPool threadPool(2);
    ProgressTracking progress;
    EXPECT_THROW(ConfocalModality::instance().generate(model, cfg, threadPool, progress),
                 std::runtime_error);
}

TEST_F(PSFGeneratorTest, GaussianGenerateRejectsNonPositiveSigma) {
    auto cfg = std::make_shared<GaussianPSFConfig>();
    cfg->sigmaX = 0.0f;
    GaussianPSFGenerator generator(cfg);
    EXPECT_THROW(generator.generatePSF(), std::runtime_error);

    auto cfg2 = std::make_shared<GaussianPSFConfig>();
    cfg2->qualityFactor = -1.0f;
    GaussianPSFGenerator generator2(cfg2);
    EXPECT_THROW(generator2.generatePSF(), std::runtime_error);
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalConfigFromJSON) {
    auto& factory = PSFGeneratorFactory::getInstance();

    auto config = factory.createConfig(confocalGlJSON(true));
    auto* gl = dynamic_cast<GibsonLanniPSFConfig*>(config.get());
    ASSERT_NE(gl, nullptr);
    EXPECT_EQ(gl->getModalityName(), "confocal");
    // The confocal parameters live on the modality config object.
    const ConfocalModalityConfig* modalityCfg = gl->modality<ConfocalModalityConfig>();
    ASSERT_NE(modalityCfg, nullptr);
    EXPECT_FLOAT_EQ(modalityCfg->excitationWavelength_nm, 488.0f);
    EXPECT_FLOAT_EQ(modalityCfg->pinholeRadius_um, 0.28f);
    EXPECT_FLOAT_EQ(modalityCfg->emissionWavelength_nm, 520.0f);

    // Copy semantics carry the modality config (setConfig copies the config).
    GibsonLanniPSFConfig copied = *gl;
    EXPECT_EQ(copied.getModalityName(), "confocal");
    const ConfocalModalityConfig* copiedModalityCfg = copied.modality<ConfocalModalityConfig>();
    ASSERT_NE(copiedModalityCfg, nullptr);
    EXPECT_FLOAT_EQ(copiedModalityCfg->excitationWavelength_nm, 488.0f);
    EXPECT_FLOAT_EQ(copiedModalityCfg->pinholeRadius_um, 0.28f);
    // The copy must own its modality config, not share the pointer.
    EXPECT_NE(copiedModalityCfg, modalityCfg);

    // Config without a modality -> widefield default, and the legacy
    // "wavelength_nm" JSON key still lands on the optical model wavelength.
    auto legacy = factory.createConfig(json::parse(TestUtils::gibsonLanniPSFConfigJSON()));
    auto* glLegacy = dynamic_cast<GibsonLanniPSFConfig*>(legacy.get());
    ASSERT_NE(glLegacy, nullptr);
    EXPECT_EQ(glLegacy->getModalityName(), "widefield");
    EXPECT_FLOAT_EQ(glLegacy->wavelength_nm, 450.0f);
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalNarrowerThanWidefield) {
    auto& factory = PSFGeneratorFactory::getInstance();

    auto widefieldConfig = factory.createConfig(confocalGlJSON(false));
    PSF widefield = generateTestPSF(widefieldConfig);
    auto confocalConfig = factory.createConfig(confocalGlJSON(true));
    PSF confocal = generateTestPSF(confocalConfig);

    // The confocal effective resolution (excitation x detection product) is
    // smaller than the widefield one, so the autosize box contracts; it must
    // never exceed the widefield box.
    EXPECT_LE(confocal.getShape().width, widefield.getShape().width);
    EXPECT_LE(confocal.getShape().depth, widefield.getShape().depth);

    ConfocalMetrics wf = measureMetrics(widefield, 100.0, 100.0);
    ConfocalMetrics cf = measureMetrics(confocal, 100.0, 100.0);

    // Unit flux for both: the confocal PSF concentrates the same energy.
    double sumWf = 0.0, sumCf = 0.0;
    for (auto it = widefield.cbegin(); it != widefield.cend(); ++it) sumWf += *it;
    for (auto it = confocal.cbegin(); it != confocal.cend(); ++it) sumCf += *it;
    EXPECT_NEAR(sumWf, 1.0, 1e-4);
    EXPECT_NEAR(sumCf, 1.0, 1e-4);

    // Confocal: narrower laterally and axially, higher peak, more in-focus energy.
    EXPECT_LT(cf.lateralFWHM_nm, wf.lateralFWHM_nm);
    EXPECT_LT(cf.axialFWHM_nm, wf.axialFWHM_nm);
    EXPECT_GT(cf.peak, 3.0 * wf.peak);
    EXPECT_GT(cf.central3Fraction, wf.central3Fraction);
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalFixedSize) {
    // The fixed-size generator branches on confocal too; both paths must
    // produce a centered, finite confocal slice stack.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = confocalGlJSON(true);
    j["size_x"] = 33;
    j["size_y"] = 33;
    j["size_z"] = 21;
    auto config = factory.createConfig(j);
    PSF psf = generateTestPSF(config);

    CuboidShape shape = psf.getShape();
    EXPECT_EQ(shape, CuboidShape(33, 33, 21));

    float peak = psf.getMax();
    EXPECT_GT(peak, 0.0f);
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    float center = psf.getPixel(cx, cy, cz);
    EXPECT_GT(center, 0.5f * peak);  // peak on the central voxel region

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalCacheInvalidation) {
    // Profiles are cached per (ti_nm, lambda): two confocal configs that
    // share ti_nm but differ only in the excitation wavelength must produce
    // different PSFs. Fixed sizes: with autosize the different excitation
    // wavelengths would also change the effective-resolution box, which is
    // not what this test is about.
    auto& factory = PSFGeneratorFactory::getInstance();
    auto buildConfig = [&factory](double excitation_nm) {
        json j = confocalGlJSON(true);
        j["excitation_wavelength_nm"] = excitation_nm;
        j["size_x"] = 33;
        j["size_y"] = 33;
        j["size_z"] = 21;
        return factory.createConfig(j);
    };

    PSF first = generateTestPSF(buildConfig(488.0));
    PSF second = generateTestPSF(buildConfig(505.0));

    EXPECT_EQ(first.getShape(), second.getShape());
    EXPECT_FALSE(first.isEqual(second, 1e-4f));
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalMetrics) {
    // User's confocal setup: NA 0.55 glycerol immersion (1.43) on water sample
    // (1.338), 488 excitation / 520 emission. Sanity ratios are asserted;
    // absolute values are printed for manual comparisoon
    // (measured: lateral 386 nm, axial 3067 nm,
    // central-3-slice energy 32%). Note: central3Fraction depends on the
    // axial pitch (100 nm here). Expected product-model values at this pitch: lateral ~338 nm,
    // axial ~2900 nm.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = confocalGlJSON(true);
    j["NA"] = 0.55;
    j["immersion_ri_design"] = 1.43;
    j["immersion_ri_experimental"] = 1.43;
    j["coverslip_thickness_design_nm"] = 170000.0;
    j["coverslip_thickness_experimental_nm"] = 170000.0;
    j["working_distance_design_nm"] = 0.0;
    j["working_distance_experimental_nm"] = 0.0;
    j["size_z"] = 32;  // imaged defocus range driving the auto-size box
    auto confocalConfig = factory.createConfig(j);
    PSF confocal = generateTestPSF(confocalConfig);

    j["microscope_modality"] = "widefield";
    auto widefieldConfig = factory.createConfig(j);
    PSF widefield = generateTestPSF(widefieldConfig);

    ConfocalMetrics wf = measureMetrics(widefield, 100.0, 100.0);
    ConfocalMetrics cf = measureMetrics(confocal, 100.0, 100.0);

    std::cout << "[ConfocalMetrics] widefield: lateralFWHM=" << wf.lateralFWHM_nm
              << "nm axialFWHM=" << wf.axialFWHM_nm
              << "nm central3=" << wf.central3Fraction * 100.0 << "%"
              << " peak=" << wf.peak << std::endl;
    std::cout << "[ConfocalMetrics] confocal:  lateralFWHM=" << cf.lateralFWHM_nm
              << "nm axialFWHM=" << cf.axialFWHM_nm
              << "nm central3=" << cf.central3Fraction * 100.0 << "%"
              << " peak=" << cf.peak
              << " peakRatio=" << cf.peak / wf.peak << std::endl;

    EXPECT_GT(cf.peak, 3.0 * wf.peak);
    EXPECT_LT(cf.lateralFWHM_nm, wf.lateralFWHM_nm);
    EXPECT_LT(cf.axialFWHM_nm, wf.axialFWHM_nm);
    EXPECT_GT(cf.lateralFWHM_nm / wf.lateralFWHM_nm, 0.55);
    EXPECT_LT(cf.lateralFWHM_nm / wf.lateralFWHM_nm, 0.95);
    EXPECT_GT(cf.axialFWHM_nm / wf.axialFWHM_nm, 0.55);
    EXPECT_LT(cf.axialFWHM_nm / wf.axialFWHM_nm, 0.95);
    EXPECT_GT(cf.central3Fraction, wf.central3Fraction);
}

TEST_F(PSFGeneratorTest, GibsonLanniPinholeZeroIsPointPinhole) {
    // pinhole_radius_um == 0 must leave the detection profile untouched, i.e.
    // the point-pinhole product model. This is the reference the finite
    // pinhole convolution is compared against.
    auto& factory = PSFGeneratorFactory::getInstance();

    json j = confocalGlJSON(true, 0.0);
    j["size_x"] = 41;
    j["size_y"] = 41;
    j["size_z"] = 9;
    j["pixel_size_axial_nm"] = 300.0;
    auto config = factory.createConfig(j);
    PSF psf = generateTestPSF(config);

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_GE(*it, 0.0f);
    }
    EXPECT_GT(psf.getMax(), 0.0f);
}

TEST_F(PSFGeneratorTest, GibsonLanniFinitePinholeBroadensDetection) {
    // A finite pinhole convolves the detection PSF with the pinhole disc,
    // which can only broaden the confocal PSF laterally relative to the
    // point-pinhole limit, while staying narrower than the widefield PSF.
    auto& factory = PSFGeneratorFactory::getInstance();

    auto build = [&](double pinhole_um) {
        json j = confocalGlJSON(true, pinhole_um);
        j["size_x"] = 41;
        j["size_y"] = 41;
        j["size_z"] = 9;
        j["pixel_size_lateral_nm"] = 50.0;
        j["pixel_size_axial_nm"] = 300.0;
        return factory.createConfig(j);
    };

    PSF point = generateTestPSF(build(0.0));
    PSF finite = generateTestPSF(build(0.3));

    ConfocalMetrics pointM = measureMetrics(point, 50.0, 300.0);
    ConfocalMetrics finiteM = measureMetrics(finite, 50.0, 300.0);

    std::cout << "[Pinhole] point lateralFWHM=" << pointM.lateralFWHM_nm
              << "nm peak=" << pointM.peak << std::endl;
    std::cout << "[Pinhole] finite(r=0.3um) lateralFWHM=" << finiteM.lateralFWHM_nm
              << "nm peak=" << finiteM.peak << std::endl;

    // Finite pinhole: broader than point pinhole, lower peak.
    EXPECT_GT(finiteM.lateralFWHM_nm, pointM.lateralFWHM_nm);
    EXPECT_LT(finiteM.peak, pointM.peak);

    // The convolution redistributes energy without creating/destroying it:
    // the integrated detection signal is preserved up to the pinhole truncation
    // at the box edge, so both stacks must be finite and non-negative.
    for (auto it = finite.cbegin(); it != finite.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_GE(*it, 0.0f);
    }
}
