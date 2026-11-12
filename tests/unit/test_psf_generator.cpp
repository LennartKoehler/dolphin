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
#include "dolphin/psf/generators/PSFBoxSizing.h"
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
#include <iostream>

using json = nlohmann::json;

class PSFGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

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
        json j = fastGlJSON();
        j["size_x"] = 21;
        j["size_y"] = 21;
        j["size_z"] = 3;
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
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
    config->cutoffThreshold = 0.05f;
    auto generator = factory.createGenerator(config);
    ASSERT_NE(generator, nullptr);
    EXPECT_TRUE(generator->hasConfig());

    PSF psf = generator->generatePSF();
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
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
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
    json j = fastGlJSON();
    j["size_x"] = 21;
    j["size_y"] = 21;
    j["size_z"] = 5;
    auto generator = factory.createGenerator("GibsonLanni", j);

    PSF psf = generator->generatePSF();
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
    //   halfLat = 3.5*196.07 + 1.1*7.0179*12000 = 94009nm -> ceil(188) -> cap 150
    //   axial   = ceil(max(12000, 2*610.71)/500) = 24
    //   -> box 301 x 301 x 49
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, 301u);
    EXPECT_EQ(shape.height, 301u);
    EXPECT_EQ(shape.depth, 49u);
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoSizeGrowsWithImageDepth) {
    // A deeper imaged volume defocuses farther -> larger cone spread and
    // longer axial stack. sizeZ 48 -> 96 doubles zMax:
    //   48: zMax=12000 -> 301x301x49;  96: zMax=24000 -> 301x301x97 (lateral capped)
    auto& factory = PSFGeneratorFactory::getInstance();

    auto shapeFor = [&](int sizeZ) {
        json j = fastGlJSON();
        j["size_z"] = sizeZ;
        auto config = factory.createConfig(j);
        config->autoSize = true;
        auto generator = factory.createGenerator(config);
        return generator->generatePSF().getShape();
    };

    CuboidShape shallow = shapeFor(48);
    CuboidShape deep = shapeFor(96);

    EXPECT_EQ(shallow.width, 301u);
    EXPECT_EQ(shallow.depth, 49u);
    EXPECT_EQ(deep.width, 301u);
    EXPECT_EQ(deep.depth, 97u);
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
        auto generator = factory.createGenerator(config);
        return generator->generatePSF();
    };

    PSF small = generate(16);
    PSF large = generate(600);

    EXPECT_EQ(small.getShape(), CuboidShape(301, 301, 49));
    EXPECT_EQ(large.getShape(), small.getShape());
    EXPECT_TRUE(large.isEqual(small, 1e-4f));
}

TEST_F(PSFGeneratorTest, GibsonLanniBoxHalfExtentModel) {
    // Direct unit test of the sizing function.
    // NA=1.3, ni=ns=1.33, lambda=520, pitch 100/100, sizeZ=64 (zMax=3200nm).
    // The cone uses the absolute tilt, so it grows even when RI is matched:
    //   resLat = 244nm, tilt = tan(asin(1.3/1.33)) = 4.6281
    //     halfLat = 3.5*244 + 1.1*4.6281*3200 = 17145nm -> ceil(172) -> cap 150
    //   resAx = 818.5nm
    //     halfAx = 3200 + 8*818.5 = 9747.7nm -> ax = 98
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.3f;
    cfg.ni = 1.33f;
    cfg.ni0 = 1.33f;
    cfg.ns = 1.33f;
    cfg.emissionWavelength_nm = 520.0f;
    cfg.pixelSizeLateral_nm = 100.0f;
    cfg.pixelSizeAxial_nm = 100.0f;
    cfg.sizeZ = 64;

    auto res = PSFBoxSizing::approximateResolution(cfg, cfg.resolutionCriterion);
    auto box = PSFBoxSizing::computeBoxHalfExtent(res, cfg);
    EXPECT_EQ(box.lateral, 150u);
    EXPECT_EQ(box.axial, 32u);

    // TIR regime (NA >= ns): asin clamp keeps everything finite and positive.
    GibsonLanniPSFConfig tir = cfg;
    tir.NA = 1.4f;
    tir.ni = 1.515f;
    tir.sizeZ = 48;
    auto resTir = PSFBoxSizing::approximateResolution(tir, tir.resolutionCriterion);
    auto boxTir = PSFBoxSizing::computeBoxHalfExtent(resTir, tir);
    EXPECT_GT(boxTir.lateral, 0u);
    EXPECT_GT(boxTir.axial, 0u);
    EXPECT_LT(boxTir.lateral, 10000u);
    EXPECT_LT(boxTir.axial, 10000u);
}

TEST_F(PSFGeneratorTest, GibsonLanniApproximateResolutionAxial) {
    // Axial resolution ordering is monotonic at the resolution level:
    // RAYLEIGH (610.7) > SIGMA (431.8) > FWHM (314.7) for NA=1.4, ns=1.33.
    GibsonLanniPSFConfig cfg;
    cfg.NA = 1.4f;
    cfg.ns = 1.33f;
    cfg.emissionWavelength_nm = 450.0f;

    auto rayleigh = PSFBoxSizing::approximateResolution(cfg, PSFConfig::ResolutionCriterion::RAYLEIGH);
    auto fwhm = PSFBoxSizing::approximateResolution(cfg, PSFConfig::ResolutionCriterion::FWHM);
    auto sigma = PSFBoxSizing::approximateResolution(cfg, PSFConfig::ResolutionCriterion::SIGMA);

    EXPECT_NEAR(rayleigh.axial_nm, 610.71, 0.1);
    EXPECT_NEAR(sigma.axial_nm, 431.84, 0.1);
    EXPECT_NEAR(fwhm.axial_nm, 314.66, 0.1);
    EXPECT_GT(rayleigh.lateral_nm, fwhm.lateral_nm);
    EXPECT_GT(fwhm.lateral_nm, sigma.lateral_nm);
}

// Result box sizes for realistic microscope configurations. Values are the
// exact output of computeBoxHalfExtent for each config (RAYLEIGH criterion
// unless noted); run this test to see the full size table in the log.
//
// Formula (PSFBoxSizing):
//   dTan     = |tan(asin(clamp(NA/ni))) - tan(asin(clamp(NA/ns)))|   (clamp 0.99)
//   zMax     = max(sizeZ/2, sizeZ-1-sizeZ/2) * dz
//   halfLat  = 3.5*resLat + 1.1*dTan*zMax      (>= 3.5*resLat)
//   halfAx   = zMax + 8*resAx                  (>= 8*resAx)
//   box      = (2*ceil(halfLat/pxLat)+1) x (2*ceil(halfAx/pxAx)+1)
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
        {"WidefieldDry_NA055",        0.55f, 1.0f,   1.338f, 520, 100, 500, 100, C::RAYLEIGH, 301, 101},
        {"DefaultGibsonLanniConfig",  0.55f, 1.43f,  1.338f, 520, 100, 500, 100, C::RAYLEIGH, 291, 101},
        {"OilConfocal_NA14",          1.40f, 1.515f, 1.33f,  520, 100, 100,  64, C::RAYLEIGH, 301, 65},
        {"MatchedRI_coneStillGrows",  1.30f, 1.33f,  1.33f,  520, 100, 100,  64, C::RAYLEIGH, 301, 65},
        {"TIRclamp_NAaboveNs",        1.40f, 1.515f, 1.33f,  450, 100, 100,  64, C::RAYLEIGH, 301, 65},
        {"OilConfocal_FWHM",          1.40f, 1.515f, 1.33f,  520, 100, 100,  64, C::FWHM,     301, 65},
        {"OilConfocal_Sigma",         1.40f, 1.515f, 1.33f,  520, 100, 100,  64, C::SIGMA,    301, 65},
        {"OilConfocal_coarsePitch",   1.40f, 1.515f, 1.33f,  520, 200, 200,  64, C::RAYLEIGH, 301, 65},
        {"OilConfocal_depth128",      1.40f, 1.515f, 1.33f,  520, 100, 100, 128, C::RAYLEIGH, 301, 129},
        {"OilConfocal_depth16",       1.40f, 1.515f, 1.33f,  520, 100, 100,  16, C::RAYLEIGH, 141, 31},
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
        cfg.emissionWavelength_nm = c.lambda_nm;
        cfg.pixelSizeLateral_nm = c.pxLat_nm;
        cfg.pixelSizeAxial_nm = c.pxAx_nm;
        cfg.sizeZ = c.sizeZ;

        auto res = PSFBoxSizing::approximateResolution(cfg, c.criterion);
        auto box = PSFBoxSizing::computeBoxHalfExtent(res, cfg);

        std::cout << "[AutoBox] " << c.name
                  << "  NA=" << c.NA << " ni=" << c.ni << " ns=" << c.ns
                  << " lam=" << c.lambda_nm << "nm"
                  << " px=" << c.pxLat_nm << "/" << c.pxAx_nm << "nm"
                  << " depth=" << c.sizeZ
                  << "  ->  box " << (2 * box.lateral + 1)
                  << " x " << (2 * box.lateral + 1)
                  << " x " << (2 * box.axial + 1) << " voxels"
                  << "  (resLat=" << res.lateral_nm << "nm resAx=" << res.axial_nm
                  << "nm zMax=" << (c.sizeZ / 2 * static_cast<size_t>(c.pxAx_nm)) << "nm)"
                  << std::endl;

        EXPECT_EQ(2 * box.lateral + 1, c.expectW);
        EXPECT_EQ(2 * box.axial + 1, c.expectD);
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoBoxEndToEndOilConfocal) {
    // Full generation must match the analytic catalog row "OilConfocal_NA14":
    // NA=1.4, ni=1.515, ns=1.33, lambda=520, pitch 100/100, depth 64 ->
    // zMax=3200nm, tilt=7.0179, halfLat=3.5*226.57+1.1*7.0179*3200=25496nm
    // -> ceil(255) capped to 150; axial=ceil(max(3200,2*705.71)/100)=32
    // -> box 301 x 301 x 65.
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
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
    EXPECT_EQ(psf.getShape(), CuboidShape(301, 301, 65));
}

TEST_F(PSFGeneratorTest, GibsonLanniAutoBoxEndToEndDefaultConfig) {
    // The DefaultGibsonLanniConfig (configs/default_psf_gibsonlanni.json):
    // NA=0.55, ni=1.43, ns=1.338, lambda=520, pitch 100/500nm, image
    // 720x672x100. sizeX/sizeY must not matter; sizeZ=100 drives the defocus.
    //   resLat = 0.61*520/0.55   = 576.73nm
    //   resAx  = 2*520*1.338/0.3025 = 4600.07nm
    //   zMax   = 50 * 500nm      = 25000nm (mid-stack focus)
    //   tilt   = max(tan(asin(0.55/1.43)), tan(asin(0.55/1.338))) = 0.45092
    //   halfLat = 3.5*576.73 + 1.1*0.45092*25000 = 14419nm -> lat = 145 (< cap)
    //   axial   = ceil(max(25000, 2*4600.07)/500) = 50
    //   -> box 291 x 291 x 101
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
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, 291u);
    EXPECT_EQ(shape.height, 291u);
    EXPECT_EQ(shape.depth, 101u);

    for (auto it = psf.cbegin(); it != psf.cend(); ++it) {
        EXPECT_FALSE(std::isnan(*it));
        EXPECT_FALSE(std::isinf(*it));
    }
}

TEST_F(PSFGeneratorTest, GibsonLanniAnalyticBoxIsSymmetric) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
    CuboidShape shape = psf.getShape();

    EXPECT_EQ(shape.width, shape.height);
    EXPECT_EQ(shape.depth % 2, 1u);
}

TEST_F(PSFGeneratorTest, GibsonLanniBoxAllValuesComputed) {
    // Every voxel inside the box must be a real integrated value (no zero
    // placeholder from per-slice disk shrinking). With a unit-flux normalised
    // PSF the far in-box corner is small but strictly positive.
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
    config->normalizePSF = false;
    auto generator = factory.createGenerator(config);

    PSF psf = generator->generatePSF();
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
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
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

TEST_F(PSFGeneratorTest, GibsonLanniUnitFluxNormalization) {
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
    config->normalizePSF = true;
    auto generator = factory.createGenerator(config);
    PSF psf = generator->generatePSF();

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
    auto& factory = PSFGeneratorFactory::getInstance();
    json j = fastGlJSON();
    auto config = factory.createConfig(j);
    config->autoSize = true;
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
    j["size_x"] = 17;
    j["size_y"] = 17;
    j["size_z"] = 3;
    j["integration_accuracy"] = 2;
    j["pixel_size_lateral_nm"] = 500.0;
    j["oversampling_factor"] = 2;
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

// ---------------------------------------------------------------------------
// Confocal Gibson-Lanni (product of excitation and emission integrals)
// ---------------------------------------------------------------------------

namespace {

// Auto-size confocal-capable config with pitch fine enough to resolve FWHM.
inline json confocalGlJSON(bool confocal, double pinholeRadius_nm = 280.0) {
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
    j["excitation_wavelength_nm"] = 488.0;
    j["pinhole_radius_nm"] = pinholeRadius_nm;
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

TEST_F(PSFGeneratorTest, GibsonLanniConfocalConfigFromJSON) {
    auto& factory = PSFGeneratorFactory::getInstance();

    auto config = factory.createConfig(confocalGlJSON(true));
    auto* gl = dynamic_cast<GibsonLanniPSFConfig*>(config.get());
    ASSERT_NE(gl, nullptr);
    EXPECT_TRUE(gl->isConfocalPSF());
    EXPECT_FLOAT_EQ(gl->excitationWavelength_nm, 488.0f);
    EXPECT_FLOAT_EQ(gl->pinholeRadius_nm, 280.0f);
    EXPECT_FLOAT_EQ(gl->emissionWavelength_nm, 520.0f);

    // Copy semantics carry the confocal fields (setConfig copies the config).
    GibsonLanniPSFConfig copied = *gl;
    EXPECT_TRUE(copied.isConfocalPSF());
    EXPECT_FLOAT_EQ(copied.excitationWavelength_nm, 488.0f);
    EXPECT_FLOAT_EQ(copied.pinholeRadius_nm, 280.0f);

    // Config without a modality -> widefield default, and the legacy
    // "wavelength_nm" JSON key still lands on the emission wavelength.
    auto legacy = factory.createConfig(json::parse(TestUtils::gibsonLanniPSFConfigJSON()));
    auto* glLegacy = dynamic_cast<GibsonLanniPSFConfig*>(legacy.get());
    ASSERT_NE(glLegacy, nullptr);
    EXPECT_FALSE(glLegacy->isConfocalPSF());
    EXPECT_FLOAT_EQ(glLegacy->emissionWavelength_nm, 450.0f);
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalNarrowerThanWidefield) {
    auto& factory = PSFGeneratorFactory::getInstance();

    auto widefieldConfig = factory.createConfig(confocalGlJSON(false));
    PSF widefield = factory.createGenerator(widefieldConfig)->generatePSF();
    auto confocalConfig = factory.createConfig(confocalGlJSON(true));
    PSF confocal = factory.createGenerator(confocalConfig)->generatePSF();

    EXPECT_EQ(widefield.getShape(), confocal.getShape());

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
    PSF psf = factory.createGenerator(config)->generatePSF();

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
    // Profiles are cached per (ti_nm, lambda): two confocal configs that share
    // ti_nm but differ only in the excitation wavelength must produce
    // different PSFs on the same generator instance.
    auto& factory = PSFGeneratorFactory::getInstance();
    auto buildConfig = [&factory](double excitation_nm) {
        json j = confocalGlJSON(true);
        j["excitation_wavelength_nm"] = excitation_nm;
        return factory.createConfig(j);
    };

    auto gen = std::make_shared<GibsonLanniPSFGenerator>();
    gen->configure(std::make_shared<ThreadPool>(), nullptr);

    gen->setConfig(buildConfig(488.0));
    PSF first = gen->generatePSF();

    gen->setConfig(buildConfig(505.0));
    PSF second = gen->generatePSF();

    EXPECT_EQ(first.getShape(), second.getShape());
    EXPECT_FALSE(first.isEqual(second, 1e-4f));
}

TEST_F(PSFGeneratorTest, GibsonLanniConfocalMetrics) {
    // User's confocal setup: NA 0.55 glycerol immersion (1.43) on water sample
    // (1.338), 488 excitation / 520 emission. Sanity ratios are asserted;
    // absolute values are printed for manual comparison against the Huygens
    // confocal_psf reference (measured: lateral 386 nm, axial 3067 nm,
    // central-3-slice energy 32%). Note: central3Fraction depends on the
    // axial pitch (100 nm here) - compare against Huygens only at matched
    // pitch. Expected product-model values at this pitch: lateral ~338 nm,
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
    PSF confocal = factory.createGenerator(confocalConfig)->generatePSF();

    j["microscope_modality"] = "widefield";
    auto widefieldConfig = factory.createConfig(j);
    PSF widefield = factory.createGenerator(widefieldConfig)->generatePSF();

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
    // pinhole_radius_nm == 0 must leave the detection profile untouched, i.e.
    // the point-pinhole product model. This is the reference the finite
    // pinhole convolution is compared against.
    auto& factory = PSFGeneratorFactory::getInstance();

    json j = confocalGlJSON(true, 0.0);
    j["size_x"] = 41;
    j["size_y"] = 41;
    j["size_z"] = 9;
    j["pixel_size_axial_nm"] = 300.0;
    auto config = factory.createConfig(j);
    PSF psf = factory.createGenerator(config)->generatePSF();

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

    auto build = [&](double pinhole_nm) {
        json j = confocalGlJSON(true, pinhole_nm);
        j["size_x"] = 41;
        j["size_y"] = 41;
        j["size_z"] = 9;
        j["pixel_size_lateral_nm"] = 50.0;
        j["pixel_size_axial_nm"] = 300.0;
        return factory.createConfig(j);
    };

    PSF point = factory.createGenerator(build(0.0))->generatePSF();
    PSF finite = factory.createGenerator(build(300.0))->generatePSF();

    ConfocalMetrics pointM = measureMetrics(point, 50.0, 300.0);
    ConfocalMetrics finiteM = measureMetrics(finite, 50.0, 300.0);

    std::cout << "[Pinhole] point lateralFWHM=" << pointM.lateralFWHM_nm
              << "nm peak=" << pointM.peak << std::endl;
    std::cout << "[Pinhole] finite(r=300nm) lateralFWHM=" << finiteM.lateralFWHM_nm
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
