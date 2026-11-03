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
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/Logging.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImagePadding.h"
#include "TestUtils.h"
#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>

class AlgorithmTestBase : public ::testing::Test {
protected:
    CuboidShape dataSize{16, 16, 16};
    IBackend* backend = nullptr;

    void SetUp() override {
        Logging::init();
        auto& factory = BackendFactory::getInstance();
        backend = &factory.getBackend<IBackend>(BackendConfig{1, "cpu"});
    }

    std::shared_ptr<DeconvolutionAlgorithm> createAlgorithm(const std::string& name) {
        DeconvolutionConfig config;
        config.algorithmName = name;
        config.iterations = 5;
        config.minDenominator = 1e-6f;
        config.signalToNoise = 0.001f;
        config.signalToNoise = 100.0f;
        auto& factory = DeconvolutionAlgorithmFactory::getInstance();
        return factory.createShared(config);
    }

    std::shared_ptr<PSF> createImpulsePSF(CuboidShape shape) {
        Image3D img(shape, 0.0f);
        img.setPixel(shape.width / 2, shape.height / 2, shape.depth / 2, 1.0f);
        return std::make_shared<PSF>(std::move(img), "impulse");
    }

    std::unique_ptr<ComplexData> preprocessPSF(std::shared_ptr<PSF> psf, CuboidShape shape) {
        ImagePadding::padToShape(*psf, shape, PaddingFillType::ZERO);
        RealData h_host = Preprocessor::convertImageToRealData(*psf);
        RealData h_device = backend->getMemoryManager().copyDataToDevice(h_host);
        std::unique_ptr<ComplexView> h_result = std::make_unique<ComplexView>(std::move(backend->getMemoryManager().reinterpret(h_device)));
        backend->getComputeManager().octantFourierShift(h_device);
        backend->getComputeManager().forwardFFT(h_device, *h_result);
        h_result->setBackend(h_device.getBackend());
        h_device.setBackend(nullptr);
        backend->sync();
        return std::move(h_result);
    }

    RealData imageToDevice(const Image3D& img) {
        RealData host = Preprocessor::convertImageToRealData(img);
        return backend->getMemoryManager().copyDataToDevice(host);
    }

    Image3D deviceToImage(RealData& data) {
        RealData host = backend->getMemoryManager().moveDataFromDevice(data, BackendFactory::getInstance().getHostBackendMemoryManager());
        return Preprocessor::convertRealDataToImage(host);
    }
};

class DeconvolutionAlgorithmTest : public AlgorithmTestBase, public ::testing::WithParamInterface<std::string> {
};

TEST_P(DeconvolutionAlgorithmTest, AlgorithmInitAndDeconvolve) {
    auto algo = createAlgorithm(GetParam());
    ASSERT_NE(algo, nullptr);

    algo->setBackend(*backend);
    algo->init(dataSize);
    EXPECT_TRUE(algo->isInitialized());

    auto psf = createImpulsePSF(dataSize);
    auto H = preprocessPSF(psf, dataSize);

    Image3D inputImg = TestUtils::createRandomImage(16, 16, 16);
    auto g = imageToDevice(inputImg);
    RealData f = backend->getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    EXPECT_NO_THROW(algo->deconvolve(*H, g, f));

    auto result = deviceToImage(f);
    EXPECT_EQ(result.getShape(), dataSize);
    EXPECT_FALSE(TestUtils::hasNaN(result));
    EXPECT_FALSE(TestUtils::hasInf(result));
}

TEST_P(DeconvolutionAlgorithmTest, ImpulsePSFReturnsInput) {
    if (GetParam() == "RichardsonLucywithAdaptiveDamping") {
        GTEST_SKIP() << "RLAD applies multiplicative damping that prevents exact convergence with impulse PSF";
    }

    auto algo = createAlgorithm(GetParam());
    ASSERT_NE(algo, nullptr);

    algo->setBackend(*backend);
    algo->init(dataSize);

    auto psf = createImpulsePSF(dataSize);
    auto H = preprocessPSF(psf, dataSize);

    Image3D inputImg = TestUtils::createRandomImage(16, 16, 16);
    auto g = imageToDevice(inputImg);
    RealData f = backend->getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    algo->deconvolve(*H, g, f);
    auto result = deviceToImage(f);

    EXPECT_TRUE(result.isEqual(inputImg, 0.1f))
        << "With impulse PSF, output should approximate input for algorithm: " << GetParam();
}

TEST_P(DeconvolutionAlgorithmTest, ZeroInputNoNaN) {
    auto algo = createAlgorithm(GetParam());
    ASSERT_NE(algo, nullptr);

    algo->setBackend(*backend);
    algo->init(dataSize);

    auto psf = createImpulsePSF(dataSize);
    auto H = preprocessPSF(psf, dataSize);

    Image3D zeroImg(dataSize, 0.0f);
    auto g = imageToDevice(zeroImg);
    RealData f = backend->getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    EXPECT_NO_THROW(algo->deconvolve(*H, g, f));
    auto result = deviceToImage(f);
    EXPECT_FALSE(TestUtils::hasNaN(result));
}

INSTANTIATE_TEST_SUITE_P(
    AllAlgorithms,
    DeconvolutionAlgorithmTest,
    ::testing::Values(
        "RichardsonLucy",
        "RichardsonLucyTotalVariation",
        "InverseFilter",
        "Convolution",
        "RichardsonLucywithAdaptiveDamping",
        "PoissonMLE"
    )
);

TEST_F(AlgorithmTestBase, RLZeroIterationsReturnsInput) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    config.iterations = 0;
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);

    algo->setBackend(*backend);
    algo->init(dataSize);

    auto psf = createImpulsePSF(dataSize);
    auto H = preprocessPSF(psf, dataSize);

    Image3D inputImg = TestUtils::createRandomImage(16, 16, 16);
    auto g = imageToDevice(inputImg);
    RealData f = backend->getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    algo->deconvolve(*H, g, f);
    auto result = deviceToImage(f);
    EXPECT_TRUE(result.isEqual(inputImg, 1.0f));
}

TEST_F(AlgorithmTestBase, ConvolutionIsForwardModel) {
    auto algo = createAlgorithm("Convolution");
    algo->setBackend(*backend);
    algo->init(dataSize);

    auto psf = createImpulsePSF(dataSize);
    auto H = preprocessPSF(psf, dataSize);

    Image3D inputImg = TestUtils::createConstantImage(16, 16, 16, 1.0f);
    auto g = imageToDevice(inputImg);
    RealData f = backend->getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    algo->deconvolve(*H, g, f);
    auto result = deviceToImage(f);

    for (auto it = result.cbegin(); it != result.cend(); ++it) {
        EXPECT_NEAR(*it, 1.0f, 0.1f);
    }
}

// Verifies the PSFHandler.cpp:183 use case: octantFourierShift moves the PSF
// peak from the image center to the (0,0,0) origin so that the subsequent
// forward FFT produces a centered OTF.
TEST_F(AlgorithmTestBase, OctantShiftMovesPSFPeakToOrigin) {
    CuboidShape shape{16, 16, 16};
    auto psf = createImpulsePSF(shape);
    ASSERT_NE(psf, nullptr);

    ImagePadding::padToShape(*psf, shape, PaddingFillType::ZERO);
    RealData h_host = Preprocessor::convertImageToRealData(*psf);
    RealData h_device = backend->getMemoryManager().copyDataToDevice(h_host);

    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    EXPECT_NEAR(h_device[centerIdx], 1.0f, 1e-5f) << "PSF peak should start at center";
    EXPECT_NEAR(h_device[0], 0.0f, 1e-5f) << "Origin should be zero before shift";

    backend->getComputeManager().octantFourierShift(h_device);
    backend->sync();

    EXPECT_NEAR(h_device[0], 1.0f, 1e-5f) << "PSF peak should be at origin after shift";
    EXPECT_NEAR(h_device[centerIdx], 0.0f, 1e-5f) << "Center should be zero after shift";
}

// Same as above but with odd dimensions — the peak at floor(N/2) must still
// reach the origin.  The last z-slice (z=D-1) is a fixed point of the shift;
// for a zero-padded PSF this is benign.
TEST_F(AlgorithmTestBase, OctantShiftMovesPSFPeakToOriginOddDims) {
    CuboidShape shape{9, 7, 5};
    auto psf = createImpulsePSF(shape);
    ASSERT_NE(psf, nullptr);

    ImagePadding::padToShape(*psf, shape, PaddingFillType::ZERO);
    RealData h_host = Preprocessor::convertImageToRealData(*psf);
    RealData h_device = backend->getMemoryManager().copyDataToDevice(h_host);

    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    EXPECT_NEAR(h_device[centerIdx], 1.0f, 1e-5f);

    backend->getComputeManager().octantFourierShift(h_device);
    backend->sync();

    EXPECT_NEAR(h_device[0], 1.0f, 1e-5f)
        << "PSF peak should be at origin after shift for odd dimensions";
}

TEST_F(AlgorithmTestBase, AlgorithmSetProgressTracker) {
    auto algo = createAlgorithm("RichardsonLucy");
    int progressCount = 0;
    algo->setProgressTracker([&progressCount](int) { progressCount++; });
    algo->setBackend(*backend);
    algo->init(dataSize);

    auto psf = createImpulsePSF(dataSize);
    auto H = preprocessPSF(psf, dataSize);

    Image3D inputImg = TestUtils::createRandomImage(16, 16, 16);
    auto g = imageToDevice(inputImg);
    RealData f = backend->getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(dataSize);

    algo->deconvolve(*H, g, f);
    EXPECT_GE(progressCount, 0);
}

TEST_F(AlgorithmTestBase, AlgorithmClonePreservesConfig) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    config.iterations = 42;
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    auto clone = algo->clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getMemoryMultiplier(), algo->getMemoryMultiplier());
}

// Regression: fillGaussianOTF must place the OTF of a separable anisotropic
// Gaussian (spatial std-devs sigmaX/Y/Z in voxels) at the correct frequency bins:
//   OTF(x,y,z) = exp(-2*pi^2 * (sx^2 fx^2 + sy^2 fy^2 + sz^2 fz^2)),
// where fx=x/W, fy=y/H (wrapped), fz=z/D (wrapped) are cycles per sample and the
// buffer is half-complex (x in [0, W/2]). The original bug used raw bin indices
// instead of cycles per sample, which for voxel sigmas collapses the OTF to a
// DC-only (image-flattening) kernel. Checking the spectrum directly avoids
// depending on the in-place real-FFT buffer layout.
TEST_F(AlgorithmTestBase, FillGaussianOTFMatchesSpatialGaussian) {
    CuboidShape shape{16, 16, 16};
    const real_t sigma = 2.0f;
    const auto& mem = backend->getMemoryManager();
    const auto& compute = backend->getComputeManager();

    const size_t W = shape.width, H = shape.height, D = shape.depth;
    const size_t complexW = W / 2 + 1;
    ComplexData otF = mem.allocateMemoryOnDeviceComplex(shape);
    compute.fillGaussianOTF(otF, sigma, sigma, sigma);
    backend->sync();

    const complex_t* p = otF.getData();
    double maxErr = 0.0;
    for (size_t z = 0; z < D; ++z)
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < complexW; ++x) {
                double fx = (double)x / (double)W;
                double fy = (y <= H / 2 ? (double)y : -(double)(H - y)) / (double)H;
                double fz = (z <= D / 2 ? (double)z : -(double)(D - z)) / (double)D;
                double expected =
                    std::exp(-2.0 * M_PI * M_PI * sigma * sigma *
                             (fx * fx + fy * fy + fz * fz));
                double got = (double)p[(z * H + y) * complexW + x][0];
                double gotIm = (double)p[(z * H + y) * complexW + x][1];
                maxErr = std::max({maxErr, std::fabs(got - expected), std::fabs(gotIm)});
            }

    // OTF values are in [0, 1]; the buggy integer-frequency version is ~e^-177
    // (== 0 in float) at the first off-DC bin, so the deviation there is ~1.0.
    EXPECT_LT(maxErr, 1e-4) << "Gaussian OTF deviates from the analytic spectrum by " << maxErr;
}

// Regression: full PoissonMLE pass on blurred data with a Gaussian PSF whose
// on-disk scale is NOT unit-sum (the algorithm must normalize it). This exercises
// the sieve (which the impulse-PSF parameter tests cannot, since the sieve sigma
// collapses to ~0 there), the PSF unit-sum normalization, and the wrapped-moment
// PSF-width estimate. The ground truth g is blurred with the UNIT-SUM version of
// the same PSF, so a correct normalization recovers the true forward model.
TEST_F(AlgorithmTestBase, PoissonMLEGaussianPSFRoundTrip) {
    const float bg = 0.01f;
    const float psfScale = 7.3f; // deliberately not unit sum
    const float sigma = 1.5f;
    const size_t S = 16, C = 8;

    // Structured ground truth: a few Gaussian "fluorescent objects" on a low floor.
    // (White-noise truth would be almost entirely destroyed by the PSF blur, leaving
    // no recoverable correlation to measure.)
    struct Blob { size_t cx, cy, cz; float amp, rad; };
    const std::vector<Blob> blobs = {
        {4, 4, 4, 1.0f, 2.0f}, {11, 5, 9, 0.8f, 2.0f}, {8, 12, 4, 0.6f, 1.5f}, {3, 11, 12, 0.9f, 2.0f},
    };
    Image3D truth(CuboidShape(S, S, S), 0.05f);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                float v = 0.05f;
                for (const auto& b : blobs) {
                    double dx = (double)x - b.cx, dy = (double)y - b.cy, dz = (double)z - b.cz;
                    v += b.amp * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (b.rad * b.rad));
                }
                truth.setPixel(x, y, z, v);
            }

    // Build two Gaussian PSF images centered at (C,C,C): the on-disk (mis-scaled)
    // one the algorithm must normalize, and the unit-sum one used to generate g.
    Image3D psfImg(dataSize, 0.0f);
    Image3D unitImg(dataSize, 0.0f);
    double psfSum = 0.0;
    std::vector<float> vals(S * S * S);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                double dx = (double)x - C, dy = (double)y - C, dz = (double)z - C;
                float v = (float)(psfScale * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (sigma * sigma)));
                vals[z * S * S + y * S + x] = v;
                psfSum += v;
            }
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                float v = vals[z * S * S + y * S + x];
                psfImg.setPixel(x, y, z, v);
                unitImg.setPixel(x, y, z, v / (float)psfSum);
            }

    auto Hmis = preprocessPSF(std::make_shared<PSF>(std::move(psfImg), "gaussian"), dataSize);
    auto Hun  = preprocessPSF(std::make_shared<PSF>(std::move(unitImg), "gaussian_unit"), dataSize);

    // g = (unit PSF) * truth + bg, using the same device FFT conventions as the algorithm.
    const auto& mem = backend->getMemoryManager();
    const auto& compute = backend->getComputeManager();
    RealData truthDev = imageToDevice(truth);
    RealData g = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    RealData work = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    ComplexView workC = mem.reinterpret(work);
    mem.memCopy(truthDev, work);
    compute.forwardFFT(work, workC);
    compute.complexMultiplication(workC, *Hun, workC);
    compute.backwardFFT(workC, work);
    compute.realAffine(work, 1.0f, bg, g);
    backend->sync();

    DeconvolutionConfig config;
    config.algorithmName = "PoissonMLE";
    config.iterations = 30;
    config.signalToNoise = 20.0f;
    config.poissonBackground = bg;
    config.poissonSieve = true;
    config.poissonPrior = true;
    config.poissonSievePasses = 2;
    auto algo = DeconvolutionAlgorithmFactory::getInstance().createShared(config);
    ASSERT_NE(algo, nullptr);
    algo->setBackend(*backend);
    algo->init(dataSize);

    RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    algo->deconvolve(*Hmis, g, f);   // pass the MIS-SCALED PSF; the algorithm must normalize it
    backend->sync();

    Image3D result = deviceToImage(f);
    EXPECT_FALSE(TestUtils::hasNaN(result));
    const float corr = TestUtils::correlation(result, truth);
    // The buggy integer-frequency sieve flattened the image to its DC (constant),
    // giving ~0 correlation with truth. A correct sieve + normalized PSF recovers
    // the objects: 0.8 sits cleanly between the broken (~0) and fixed (~0.9) cases.
    EXPECT_GT(corr, 0.8f) << "Gaussian-PSF round-trip correlation with ground truth was " << corr;
}


// Regression: poissonSieveScale flows config -> algorithm, and turning the sieve off
// (scale 0 = identity) is at least as faithful to and at least as sharp as the default
// (scale 0.5). The sieve is a DC-preserving low-pass, so smoothing can only redistribute
// peak energy into the background (lower max, lower correlation) — it cannot recover
// detail. This guards the knob and the contrast behavior.
TEST_F(AlgorithmTestBase, PoissonMLESieveScaleKnob) {
    const float bg = 0.01f;
    const float psfScale = 7.3f; // deliberately not unit sum
    const float sigma = 1.5f;
    const size_t S = 16, C = 8;

    // Same structured ground truth as PoissonMLEGaussianPSFRoundTrip.
    struct Blob { size_t cx, cy, cz; float amp, rad; };
    const std::vector<Blob> blobs = {
        {4, 4, 4, 1.0f, 2.0f}, {11, 5, 9, 0.8f, 2.0f}, {8, 12, 4, 0.6f, 1.5f}, {3, 11, 12, 0.9f, 2.0f},
    };
    Image3D truth(CuboidShape(S, S, S), 0.05f);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                float v = 0.05f;
                for (const auto& b : blobs) {
                    double dx = (double)x - b.cx, dy = (double)y - b.cy, dz = (double)z - b.cz;
                    v += b.amp * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (b.rad * b.rad));
                }
                truth.setPixel(x, y, z, v);
            }

    Image3D psfImg(dataSize, 0.0f);
    Image3D unitImg(dataSize, 0.0f);
    double psfSum = 0.0;
    std::vector<float> vals(S * S * S);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                double dx = (double)x - C, dy = (double)y - C, dz = (double)z - C;
                float v = (float)(psfScale * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (sigma * sigma)));
                vals[z * S * S + y * S + x] = v;
                psfSum += v;
            }
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                float v = vals[z * S * S + y * S + x];
                psfImg.setPixel(x, y, z, v);
                unitImg.setPixel(x, y, z, v / (float)psfSum);
            }

    const auto& mem = backend->getMemoryManager();
    const auto& compute = backend->getComputeManager();
    RealData g = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    {
        RealData truthDev = imageToDevice(truth);
        RealData work = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
        ComplexView workC = mem.reinterpret(work);
        mem.memCopy(truthDev, work);
        compute.forwardFFT(work, workC);
        compute.complexMultiplication(workC, *preprocessPSF(std::make_shared<PSF>(std::move(unitImg), "gaussian_unit"), dataSize), workC);
        compute.backwardFFT(workC, work);
        compute.realAffine(work, 1.0f, bg, g);
        backend->sync();
    }
    auto Hmis = preprocessPSF(std::make_shared<PSF>(std::move(psfImg), "gaussian"), dataSize);

    // Returns Image3D by value (copy-elided, like the round-trip test above).
    // snr=1 so poissonSieveScale maps 1:1 to its literal meaning (effective sieve
    // scale = poissonSieveScale / snr); the flow-through check stays unambiguous.
    auto run = [this, &mem, &bg, &g, &Hmis](float sieveScale) -> Image3D {
        DeconvolutionConfig config;
        config.algorithmName = "PoissonMLE";
        config.iterations = 30;
        config.signalToNoise = 1.0f;
        config.poissonBackground = bg;
        config.poissonSieve = true;
        config.poissonPrior = true;
        config.poissonSievePasses = 2;
        config.poissonSieveScale = sieveScale;
        auto algo = DeconvolutionAlgorithmFactory::getInstance().createShared(config);
        if (!algo) return Image3D(dataSize, 0.0f);
        algo->setBackend(*backend);
        algo->init(dataSize);
        RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
        algo->deconvolve(*Hmis, g, f);
        backend->sync();
        return deviceToImage(f);
    };

    Image3D resOff = run(0.0f);    // sieve disabled (identity)
    Image3D resDef = run(0.5f);    // original sieve strength

    // 1) Stability: the solver must stay finite with the sieve turned off.
    EXPECT_FALSE(TestUtils::hasNaN(resOff));
    EXPECT_FALSE(TestUtils::hasInf(resOff));

    // 2) Fidelity: with the sieve off the objects are still recovered well.
    const float corrOff = TestUtils::correlation(resOff, truth);
    EXPECT_GT(corrOff, 0.8f) << "sieve off (scale 0) correlation with ground truth was " << corrOff;

    // 3) Flow-through: if poissonSieveScale were not wired config -> algorithm the
    //    two runs would be identical. They must differ (the sieve changes the result).
    float maxDiff = 0.0f;
    const size_t vol = resOff.getShape().getVolume();
    for (size_t i = 0; i < vol; ++i) maxDiff = std::max(maxDiff, std::fabs(resOff[i] - resDef[i]));
    EXPECT_GT(maxDiff, 1e-4f)
        << "poissonSieveScale had no effect: sieve-off and sieve-on results are identical";
}


// Regression: the prefilter init step. With an AUTO background (poissonBackground = -1)
// and prefilter ON, the initial estimate f0 is a mild anisotropic Gaussian low-pass of
// the data and the background is min(f0); with prefilter OFF it falls back to the raw
// data and the dimmest-decile mean. Both must reconstruct the objects and stay finite,
// and the prefilter must not degrade the result.
TEST_F(AlgorithmTestBase, PoissonMLEPrefilterAutoBackground) {
    const float bg = 0.01f;
    const float sigma = 1.5f;
    const size_t S = 16, C = 8;

    struct Blob { size_t cx, cy, cz; float amp, rad; };
    const std::vector<Blob> blobs = {
        {4, 4, 4, 1.0f, 2.0f}, {11, 5, 9, 0.8f, 2.0f}, {8, 12, 4, 0.6f, 1.5f}, {3, 11, 12, 0.9f, 2.0f},
    };
    Image3D truth(CuboidShape(S, S, S), 0.05f);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                float v = 0.05f;
                for (const auto& b : blobs) {
                    double dx = (double)x - b.cx, dy = (double)y - b.cy, dz = (double)z - b.cz;
                    v += b.amp * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (b.rad * b.rad));
                }
                truth.setPixel(x, y, z, v);
            }

    Image3D psfImg(dataSize, 0.0f);
    double psfSum = 0.0;
    std::vector<float> vals(S * S * S);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                double dx = (double)x - C, dy = (double)y - C, dz = (double)z - C;
                float v = (float)std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (sigma * sigma));
                vals[z * S * S + y * S + x] = v;
                psfSum += v;
            }
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x)
                psfImg.setPixel(x, y, z, vals[z * S * S + y * S + x] / (float)psfSum);
    auto H = preprocessPSF(std::make_shared<PSF>(std::move(psfImg), "gaussian_unit"), dataSize);

    const auto& mem = backend->getMemoryManager();
    const auto& compute = backend->getComputeManager();
    RealData truthDev = imageToDevice(truth);
    RealData g = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    RealData work = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    ComplexView workC = mem.reinterpret(work);
    mem.memCopy(truthDev, work);
    compute.forwardFFT(work, workC);
    compute.complexMultiplication(workC, *H, workC);
    compute.backwardFFT(workC, work);
    compute.realAffine(work, 1.0f, bg, g);
    backend->sync();

    auto run = [&](bool prefilter) -> Image3D {
        DeconvolutionConfig config;
        config.algorithmName = "PoissonMLE";
        config.iterations = 30;
        config.signalToNoise = 20.0f;
        config.poissonBackground = -1.0f; // auto
        config.poissonPrefilter = prefilter;
        config.poissonPrefilterScale = 0.3f;
        config.poissonSieve = true;
        config.poissonPrior = true;
        config.poissonSievePasses = 2;
        auto algo = DeconvolutionAlgorithmFactory::getInstance().createShared(config);
        if (!algo) throw std::runtime_error("PoissonMLE algorithm not created");
        algo->setBackend(*backend);
        algo->init(dataSize);
        RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
        algo->deconvolve(*H, g, f);
        backend->sync();
        return deviceToImage(f);
    };

    Image3D withPrefilter = run(true);
    EXPECT_FALSE(TestUtils::hasNaN(withPrefilter));
    EXPECT_FALSE(TestUtils::hasInf(withPrefilter));
    const float corrOn = TestUtils::correlation(withPrefilter, truth);
    EXPECT_GT(corrOn, 0.7f) << "prefilter (auto bg) correlation with ground truth was " << corrOn;

    Image3D withoutPrefilter = run(false);
    EXPECT_FALSE(TestUtils::hasNaN(withoutPrefilter));
    const float corrOff = TestUtils::correlation(withoutPrefilter, truth);
    EXPECT_GT(corrOff, 0.7f) << "no-prefilter (auto bg) correlation with ground truth was " << corrOff;

    // The prefilter gives a better start, so it must not hurt reconstruction.
    EXPECT_GE(corrOn, corrOff - 0.05f)
        << "prefilter-on corr " << corrOn << " < prefilter-off corr " << corrOff;
}

// Regression: the output must be reported on a ~zero background. With an auto
// background the MLE absorbs (flat background of g) - bg into f as a uniform
// pedestal unless it is explicitly removed. Ground truth here has OBJECT
// background 0 on a flat data background bg, so the correct output background is
// ~0; this asserts the final pedestal removal leaves the mostly-background voxels
// near zero (the earlier symptom was a leftover floor ~= bg).
TEST_F(AlgorithmTestBase, PoissonMLERemovesOutputPedestal) {
    const float bg = 0.02f;              // measured (data) background
    const float sigma = 2.0f;
    const size_t S = 16, C = 8;

    // Sparse bright objects on a ZERO object background -> most voxels are flat bg.
    struct Blob { size_t cx, cy, cz; float amp, rad; };
    const std::vector<Blob> blobs = {
        {4, 4, 4, 5.0f, 1.0f}, {12, 5, 11, 5.0f, 1.0f}, {8, 12, 3, 5.0f, 1.0f},
    };
    Image3D truth(CuboidShape(S, S, S), 0.0f);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                float v = 0.0f;
                for (const auto& b : blobs) {
                    double dx = (double)x - b.cx, dy = (double)y - b.cy, dz = (double)z - b.cz;
                    v += b.amp * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (b.rad * b.rad));
                }
                truth.setPixel(x, y, z, v);
            }

    Image3D psfImg(dataSize, 0.0f);
    double psfSum = 0.0;
    std::vector<float> vals(S * S * S);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                double dx = (double)x - C, dy = (double)y - C, dz = (double)z - C;
                float v = (float)std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (sigma * sigma));
                vals[z * S * S + y * S + x] = v;
                psfSum += v;
            }
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x)
                psfImg.setPixel(x, y, z, vals[z * S * S + y * S + x] / (float)psfSum);
    auto H = preprocessPSF(std::make_shared<PSF>(std::move(psfImg), "gaussian_unit"), dataSize);

    const auto& mem = backend->getMemoryManager();
    const auto& compute = backend->getComputeManager();
    RealData truthDev = imageToDevice(truth);
    RealData g = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    RealData work = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    ComplexView workC = mem.reinterpret(work);
    mem.memCopy(truthDev, work);
    compute.forwardFFT(work, workC);
    compute.complexMultiplication(workC, *H, workC);
    compute.backwardFFT(workC, work);
    compute.realAffine(work, 1.0f, bg, g);
    backend->sync();

    DeconvolutionConfig config;
    config.algorithmName = "PoissonMLE";
    config.iterations = 30;
    config.signalToNoise = 20.0f;
    config.poissonBackground = -1.0f; // auto
    config.poissonPrefilter = true;
    config.poissonPrefilterScale = 0.3f;
    config.poissonSieve = true;
    config.poissonPrior = true;
    config.poissonSievePasses = 2;
    auto algo = DeconvolutionAlgorithmFactory::getInstance().createShared(config);
    ASSERT_NE(algo, nullptr);
    algo->setBackend(*backend);
    algo->init(dataSize);

    RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(dataSize);
    algo->deconvolve(*H, g, f);
    backend->sync();

    Image3D result = deviceToImage(f);
    EXPECT_FALSE(TestUtils::hasNaN(result));

    // The lower quartile of a mostly-background image is its flat background; it
    // must sit near zero (well below bg), not on a leftover pedestal ~= bg.
    std::vector<float> vals2;
    vals2.reserve(result.getShape().getVolume());
    for (const auto& it : result) vals2.push_back((float)it);
    std::sort(vals2.begin(), vals2.end());
    const float q25 = vals2[vals2.size() / 4];
    const float q50 = vals2[vals2.size() / 2];
    EXPECT_LT(q25, 0.03f * bg) << "output lower-quartile background " << q25 << " close to data bg " << bg;
    EXPECT_LT(q50, 0.08f * bg) << "output median background " << q50 << " close to data bg " << bg;
    // Objects still present.
    EXPECT_GT(vals2.back(), 1.0f) << "objects were removed by the pedestal subtraction";
}

// Robustness: a PSF with negative lobes (negative DC after centering) must not
// drive the estimate to NaN/Inf. Exercises the model-floor + line-search guards.
TEST_F(AlgorithmTestBase, PoissonMLENegativePSFNoNaN) {
    CuboidShape shape{17, 16, 16};
    const size_t V = shape.getVolume();
    const auto& mem = backend->getMemoryManager();

    Image3D img(shape, 0.0f);
    for (size_t z = 0; z < 16; ++z) for (size_t y = 0; y < 16; ++y) for (size_t x = 0; x < 17; ++x) {
        double dx = x - 5, dy = y - 6, dz = z - 7;
        img.setPixel(x, y, z, (float)(0.8 * std::exp(-0.5 * (dx*dx + dy*dy + dz*dz) / 2.0) +
                                    0.4 * std::exp(-0.5 * ((x-11)*(x-11) + (y-12)*(y-12) + dz*dz) / 1.2)));
    }
    // PSF with NEGATIVE DC: positive core, negative wider halo
    Image3D psfImg(shape, 0.0f);
    for (size_t z = 0; z < 16; ++z) for (size_t y = 0; y < 16; ++y) for (size_t x = 0; x < 17; ++x) {
        double dx = x - 8.0, dy = y - 8.0, dz = z - 8.0;
        double r2 = dx*dx + dy*dy + dz*dz;
        psfImg.setPixel(x, y, z, (float)(std::exp(-r2 / 0.5) - 0.6 * std::exp(-r2 / 6.0)));
    }
    auto psf = std::make_shared<PSF>(std::move(psfImg), "negpsf");
    auto H = preprocessPSF(psf, shape);

    RealData g = imageToDevice(img);
    RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(shape);

    DeconvolutionConfig config;
    config.algorithmName = "PoissonMLE";
    config.iterations = 2;
    config.signalToNoise = 10.0f;
    config.poissonSieve = false;
    config.poissonPrior = false;
    auto algo = DeconvolutionAlgorithmFactory::getInstance().createShared(config);
    algo->setBackend(*backend);
    algo->init(shape);
    algo->deconvolve(*H, g, f);
    backend->sync();

    // f is a padded FFT-in-place buffer: read it strided (via access), not flat.
    double mn = 1e300, mx = -1e300;
    size_t nan = 0, neg = 0;
    for (size_t i = 0; i < V; ++i) {
        real_t v = f[i];
        if (std::isnan(v) || std::isinf(v)) { ++nan; continue; }
        if (v < 0) ++neg;
        mn = std::min(mn, (double)v); mx = std::max(mx, (double)v);
    }
    EXPECT_EQ(nan, 0u) << "estimate contains NaN/Inf (min=" << mn << " max=" << mx << ")";
}

// Regression: a wide PSF (so the annealed sieve has a real smoothing width) on an
// odd-height work shape with sharp point sources + a flat background. The FFT-based
// sieve reconstructs the flat background only up to float32 roundoff and can dip
// below the background level (negative ringing); without a post-sieve clamp that
// negative f feeds the next data step's g/m ratio and the iteration runs away to
// inf/NaN. This test asserts the estimate stays finite and non-negative.
TEST_F(AlgorithmTestBase, PoissonMLEWidePSFSharpPeaksNoNaN) {
    // Wide PSF + flat background + sharp points: the sieve's float32 ringing leaves
    // f slightly below bg, which the model floor must absorb. Non-cubic shape
    // (32x31x24) exercises the strided FFT layout. The PSF box (21) must fit inside
    // the shape in every dimension or padToShape cannot shrink it.
    CuboidShape shape{32, 31, 24};
    const auto& mem = backend->getMemoryManager();
    const auto& compute = backend->getComputeManager();

    const real_t bg = 0.015f;
    Image3D gImg(shape, bg);
    // a few sharp point-spread "fluorescent" objects on the flat background
    const std::vector<std::array<size_t, 3>> peaks = {{8, 10, 6}, {20, 22, 9}, {14, 4, 12}};
    for (const auto& p : peaks)
        gImg.setPixel(p[0], p[1], p[2], 1.0f);

    // wide Gaussian PSF (sigma 6) -> psfLat ~ 6, sieve sigma ~ 3 at k=0
    const real_t psfSigma = 6.0f;
    const size_t S = 21;            // PSF box (odd, centered, fits in shape)
    const real_t C = (S - 1) / 2.0;
    double psfSum = 0.0;
    std::vector<float> vals(S * S * S);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x) {
                double dx = (double)x - C, dy = (double)y - C, dz = (double)z - C;
                float v = (float)std::exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (psfSigma * psfSigma));
                vals[z * S * S + y * S + x] = v;
                psfSum += v;
            }
    Image3D psfImg(CuboidShape{S, S, S}, 0.0f);
    for (size_t z = 0; z < S; ++z)
        for (size_t y = 0; y < S; ++y)
            for (size_t x = 0; x < S; ++x)
                psfImg.setPixel(x, y, z, vals[z * S * S + y * S + x] / (float)psfSum);
    auto H = preprocessPSF(std::make_shared<PSF>(std::move(psfImg), "wide"), shape);

    DeconvolutionConfig config;
    config.algorithmName = "PoissonMLE";
    config.iterations = 6;
    config.signalToNoise = 10.0f;      // lambda = 0.1
    config.poissonBackground = bg;
    config.poissonSieve = true;
    config.poissonPrior = true;
    config.poissonSievePasses = 2;
    auto algo = DeconvolutionAlgorithmFactory::getInstance().createShared(config);
    ASSERT_NE(algo, nullptr);
    algo->setBackend(*backend);
    algo->init(shape);

    RealData g = imageToDevice(gImg);
    RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(shape);
    algo->deconvolve(*H, g, f);
    backend->sync();

    Image3D result = deviceToImage(f);
    EXPECT_FALSE(TestUtils::hasNaN(result)) << "estimate contains NaN";
    EXPECT_FALSE(TestUtils::hasInf(result)) << "estimate contains Inf";
    float mn = 1e30f;
    for (const auto& it : result) mn = std::min(mn, (float)it);
    EXPECT_GE(mn, 0.0f) << "estimate went negative (min=" << mn << ")";
}
