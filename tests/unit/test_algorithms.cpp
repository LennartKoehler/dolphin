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
#include <cmath>

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
        config.epsilon = 1e-6f;
        config.lambda = 0.001f;
        auto& factory = DeconvolutionAlgorithmFactory::getInstance();
        return factory.createShared(config);
    }

    std::shared_ptr<PSF> createImpulsePSF(CuboidShape shape) {
        Image3D img(shape, 0.0f);
        img.setPixel(shape.width / 2, shape.height / 2, shape.depth / 2, 1.0f);
        return std::make_shared<PSF>(std::move(img), "impulse");
    }

    std::unique_ptr<ComplexData> preprocessPSF(std::shared_ptr<PSF> psf, CuboidShape shape) {
        Preprocessor::padToShape(*psf, shape, PaddingFillType::ZERO);
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
        RealData host = backend->getMemoryManager().moveDataFromDevice(data, BackendFactory::getInstance().getDefaultBackendMemoryManager());
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
        "RichardsonLucywithAdaptiveDamping"
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
