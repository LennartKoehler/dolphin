#include <gtest/gtest.h>
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin_image/Image3D.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"
#include <cmath>

class FFTBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(FFTBackendTest, FFTRoundTripPreservesData) {
    CuboidShape shape{16, 16, 16};
    Image3D inputImg = TestUtils::createRandomImage(16, 16, 16);

    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    RealData realData = Preprocessor::convertImageToRealData(inputImg);
    ComplexData complexData = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    compute.forwardFFT(realData, complexData);
    compute.backwardFFT(complexData, realData);

    auto outputImg = Preprocessor::convertRealDataToImage(realData);

    EXPECT_TRUE(outputImg.isEqual(inputImg, 0.05f))
        << "FFT round-trip should preserve data within tolerance";
}

TEST_F(FFTBackendTest, FFTImpulseResponse) {
    CuboidShape shape{8, 8, 8};
    Image3D impulse = TestUtils::createImpulseImage(8, 8, 8);

    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    RealData realData = Preprocessor::convertImageToRealData(impulse);
    ComplexData complexData = memMgr.allocateMemoryOnDeviceComplex(shape);

    compute.forwardFFT(realData, complexData);

    for (size_t i = 0; i < static_cast<size_t>(complexData.getSize().getVolume()); i++) {
        float mag = std::sqrt(
            complexData.access(i)[0] * complexData.access(i)[0] +
            complexData.access(i)[1] * complexData.access(i)[1]
        );
        EXPECT_NEAR(mag, 1.0f, 0.01f)
            << "FFT of impulse should have constant magnitude";
    }
}

TEST_F(FFTBackendTest, FFTConstantInput) {
    CuboidShape shape{8, 8, 8};
    Image3D constant(CuboidShape(8, 8, 8), 5.0f);

    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    RealData realData = Preprocessor::convertImageToRealData(constant);
    ComplexData complexData = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    compute.forwardFFT(realData, complexData);

    float dcMag = std::sqrt(
        complexData.access(0)[0] * complexData.access(0)[0] +
        complexData.access(0)[1] * complexData.access(0)[1]
    );
    EXPECT_GT(dcMag, 0.0f);

    for (size_t i = 1; i < static_cast<size_t>(shape.getVolume()); i++) {
        float mag = std::sqrt(
            complexData.access(i)[0] * complexData.access(i)[0] +
            complexData.access(i)[1] * complexData.access(i)[1]
        );
        EXPECT_NEAR(mag, 0.0f, 0.01f)
            << "FFT of constant should be zero except DC component";
    }
}

TEST_F(FFTBackendTest, OctantShiftIdentity) {
    CuboidShape shape{8, 8, 8};
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        data.access(i)[0] = static_cast<float>(i);
        data.access(i)[1] = static_cast<float>(i) * 0.5f;
    }

    std::vector<float> original(shape.getVolume() * 2);
    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        original[i * 2] = data.access(i)[0];
        original[i * 2 + 1] = data.access(i)[1];
    }

    compute.octantFourierShift(data);
    compute.octantFourierShift(data);

    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        EXPECT_NEAR(data.access(i)[0], original[i * 2], 0.01f);
        EXPECT_NEAR(data.access(i)[1], original[i * 2 + 1], 0.01f);
    }
}
