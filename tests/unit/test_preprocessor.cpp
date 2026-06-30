#include <gtest/gtest.h>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/Logging.h"
#include "dolphin_image/Image3D.h"
#include "TestUtils.h"

class PreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }

    IBackend& getCPUBackend() {
        return BackendFactory::getInstance().getBackend<IBackend>(BackendConfig{1, "cpu"});
    }
};

TEST_F(PreprocessorTest, ConvertImageToRealDataRoundTrip) {
    Image3D img = TestUtils::createRandomImage(8, 8, 8);
    auto realData = Preprocessor::convertImageToRealData(img);
    auto result = Preprocessor::convertRealDataToImage(realData);
    EXPECT_TRUE(result.isEqual(img, 0.001f));
}

TEST_F(PreprocessorTest, ConvertImageToComplexDataRoundTrip) {
    Image3D img = TestUtils::createRandomImage(8, 8, 8);
    auto complexData = Preprocessor::convertImageToComplexData(img);
    auto result = Preprocessor::convertComplexDataToImage(complexData);
    EXPECT_EQ(result.getShape(), img.getShape());
    EXPECT_TRUE(result.isEqual(img, 0.001f));
}

TEST_F(PreprocessorTest, ConvertConstantImage) {
    Image3D img(CuboidShape(4, 4, 4), 3.14f);
    auto realData = Preprocessor::convertImageToRealData(img);
    auto result = Preprocessor::convertRealDataToImage(realData);
    for (auto it = result.cbegin(); it != result.cend(); ++it) {
        EXPECT_NEAR(*it, 3.14f, 0.001f);
    }
}

TEST_F(PreprocessorTest, ConvertZeroImage) {
    Image3D img(CuboidShape(4, 4, 4), 0.0f);
    auto realData = Preprocessor::convertImageToRealData(img);
    auto result = Preprocessor::convertRealDataToImage(realData);
    for (auto it = result.cbegin(); it != result.cend(); ++it) {
        EXPECT_FLOAT_EQ(*it, 0.0f);
    }
}

TEST_F(PreprocessorTest, ConvertLargeImage) {
    Image3D img = TestUtils::createRandomImage(32, 32, 16);
    auto realData = Preprocessor::convertImageToRealData(img);
    auto result = Preprocessor::convertRealDataToImage(realData);
    EXPECT_TRUE(result.isEqual(img, 0.001f));
}

TEST_F(PreprocessorTest, PadImageMirrorWrapper) {
    Image3D img(CuboidShape(4, 4, 4), 1.0f);
    Padding padding;
    padding.before = CuboidShape(2, 2, 2);
    padding.after = CuboidShape(2, 2, 2);
    Preprocessor::padImageMirror(img, padding);
    EXPECT_EQ(img.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(PreprocessorTest, PadImageZeroWrapper) {
    Image3D img(CuboidShape(4, 4, 4), 1.0f);
    Padding padding;
    padding.before = CuboidShape(1, 1, 1);
    padding.after = CuboidShape(1, 1, 1);
    Preprocessor::padImageZero(img, padding);
    EXPECT_EQ(img.getShape(), CuboidShape(6, 6, 6));
    EXPECT_FLOAT_EQ(img.getPixel(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(img.getPixel(1, 1, 1), 1.0f);
}

TEST_F(PreprocessorTest, ExpandToMinSizeWrapper) {
    Image3D img(CuboidShape(4, 4, 4), 2.0f);
    Preprocessor::expandToMinSize(img, CuboidShape(8, 8, 8));
    EXPECT_EQ(img.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(PreprocessorTest, PadToShapeWrapper) {
    Image3D img(CuboidShape(4, 4, 4), 1.0f);
    auto padding = Preprocessor::padToShape(img, CuboidShape(8, 8, 8), PaddingFillType::ZERO);
    EXPECT_EQ(img.getShape(), CuboidShape(8, 8, 8));
}

TEST_F(PreprocessorTest, ParentPadding) {
    PSF psf(Image3D(CuboidShape(8, 8, 8), 0.0f), "test");
    psf.setPixel(4, 4, 4, 1.0f);
    psf.setPixel(0, 0, 0, 0.0001f);

    auto padding = PaddingStrategy::parentPadding(psf, 0.001f);
    EXPECT_GE(padding.width, 0);
    EXPECT_GE(padding.height, 0);
    EXPECT_GE(padding.depth, 0);
}

TEST_F(PreprocessorTest, FullPSFPadding) {
    PSF psf(Image3D(CuboidShape(8, 8, 8), 1.0f), "test");
    auto padding = PaddingStrategy::fullPSFPadding(psf);
    EXPECT_EQ(padding.width, 8);
    EXPECT_EQ(padding.height, 8);
    EXPECT_EQ(padding.depth, 8);
}

TEST_F(PreprocessorTest, PSFPreprocessorCaching) {
    PSFPreprocessor preprocessor;
    auto& backend = getCPUBackend();

    int callCount = 0;
    preprocessor.setPreprocessingFunction(
        [&callCount](CuboidShape, std::shared_ptr<PSF>, IBackend&) -> std::unique_ptr<ComplexData> {
            callCount++;
            return nullptr;
        }
    );

    auto psf = std::make_shared<PSF>(Image3D(CuboidShape(4, 4, 4), 1.0f), "test_psf");
    CuboidShape shape{8, 8, 8};

    preprocessor.getPreprocessedPSF(shape, psf, backend);
    preprocessor.getPreprocessedPSF(shape, psf, backend);

    EXPECT_EQ(callCount, 1);
}
