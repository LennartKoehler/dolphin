#include <gtest/gtest.h>
#include "dolphin/deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin/psf/PSF.h"
#include "dolphin_image/Image3D.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"

class TestableLabeledDeconvolutionExecutor : public LabeledDeconvolutionExecutor {
public:
    using LabeledDeconvolutionExecutor::getLabelGroups;
    using LabeledDeconvolutionExecutor::makeMasksWeighted;
    using LabeledDeconvolutionExecutor::createGaussianKernel;
};

class GetLabelGroupsTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();

        labelImage = Image3D(CuboidShape(16, 16, 4), 0.0f);
        for (size_t z = 0; z < 4; z++) {
            for (size_t y = 0; y < 16; y++) {
                for (size_t x = 0; x < 16; x++) {
                    if (x < 8) {
                        labelImage.setPixel(x, y, z, 1.0f);
                    } else {
                        labelImage.setPixel(x, y, z, 100.0f);
                    }
                }
            }
        }

        auto createDummyPSF = [](const std::string& id) -> std::shared_ptr<PSF> {
            Image3D psfImage(CuboidShape{3, 3, 3}, 1.0f);
            return std::make_shared<PSF>(std::move(psfImage), id);
        };

        psfs.clear();
        psfs.push_back(createDummyPSF("psf_a"));
        psfs.push_back(createDummyPSF("psf_b"));

        psfLabelMap.addRange(0, 2, "psf_a");
        psfLabelMap.addRange(50, 200, "psf_b");
    }

    Image3D labelImage;
    std::vector<std::shared_ptr<PSF>> psfs;
    RangeMap<std::string> psfLabelMap;
    TestableLabeledDeconvolutionExecutor executor;
};

TEST_F(GetLabelGroupsTest, CreatesLabelGroups) {
    auto labelGroups = executor.getLabelGroups(psfs, labelImage, psfLabelMap);
    EXPECT_GE(labelGroups.size(), 1u);
}

TEST_F(GetLabelGroupsTest, LabelGroupsHaveMasks) {
    auto labelGroups = executor.getLabelGroups(psfs, labelImage, psfLabelMap);
    for (size_t i = 0; i < labelGroups.size(); i++) {
        const Image3D* mask = labelGroups[i].getMask();
        EXPECT_NE(mask, nullptr);
    }
}

TEST_F(GetLabelGroupsTest, LabelGroupsHavePSFs) {
    auto labelGroups = executor.getLabelGroups(psfs, labelImage, psfLabelMap);
    for (size_t i = 0; i < labelGroups.size(); i++) {
        auto groupPsfs = labelGroups[i].getPSFs();
        EXPECT_GE(groupPsfs.size(), 1u);
    }
}

TEST_F(GetLabelGroupsTest, CreateGaussianKernel) {
    int radius = 5;
    auto kernel = executor.createGaussianKernel(radius);
    EXPECT_EQ(kernel->getShape().width, 20);
    EXPECT_EQ(kernel->getShape().height, 20);
    EXPECT_EQ(kernel->getShape().depth, 20);
}

TEST_F(GetLabelGroupsTest, GaussianKernelIsCentered) {
    auto kernel = executor.createGaussianKernel(5);
    float maxVal = kernel->getMax();
    float centerVal = kernel->getPixel(10, 10, 10);
    EXPECT_NEAR(centerVal, maxVal, maxVal * 0.01f);
}

TEST_F(GetLabelGroupsTest, GaussianKernelNoNaN) {
    auto kernel = executor.createGaussianKernel(5);
    EXPECT_FALSE(TestUtils::hasNaN(*kernel));
}
