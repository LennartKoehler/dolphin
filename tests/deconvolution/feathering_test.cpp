#include <gtest/gtest.h>
#include "dolphin/deconvolution/Postprocessor.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "TestUtils.h"

class FeatheringTest : public ::testing::Test {
protected:
    void SetUp() override {
        imageA = Image3D(CuboidShape(16, 16, 8), 0.0f);
        imageB = Image3D(CuboidShape(16, 16, 8), 0.0f);

        for (size_t z = 0; z < 8; z++) {
            for (size_t y = 0; y < 16; y++) {
                for (size_t x = 0; x < 16; x++) {
                    if (x < 8) {
                        imageA.setPixel(x, y, z, 1.0f);
                    } else {
                        imageB.setPixel(x, y, z, 1.0f);
                    }
                }
            }
        }

        maskA = imageA.getInRange(0.5f, 1.5f);
        maskB = imageB.getInRange(0.5f, 1.5f);
    }

    Image3D imageA, imageB;
    Image3D maskA, maskB;
};

TEST_F(FeatheringTest, AddFeatheringDoesNotCrash) {
    std::vector<ImageMaskPair> pairs;
    pairs.push_back({imageA, maskA});
    pairs.push_back({imageB, maskB});

    size_t radius = 5;
    float epsilon = 5.0f;
    EXPECT_NO_THROW(Postprocessor::addFeathering(pairs, radius, epsilon));
}

TEST_F(FeatheringTest, AddFeatheringProducesValidOutput) {
    std::vector<ImageMaskPair> pairs;
    pairs.push_back({imageA, maskA});
    pairs.push_back({imageB, maskB});

    auto result = Postprocessor::addFeathering(pairs, 3, 1.0f);
    EXPECT_EQ(result.getShape(), imageA.getShape());
    EXPECT_FALSE(TestUtils::hasNaN(result));
}

TEST_F(FeatheringTest, AddFeatheringSingleImage) {
    std::vector<ImageMaskPair> pairs;
    pairs.push_back({imageA, maskA});

    auto result = Postprocessor::addFeathering(pairs, 3, 1.0f);
    EXPECT_EQ(result.getShape(), imageA.getShape());
}

TEST_F(FeatheringTest, AddFeatheringZeroRadius) {
    std::vector<ImageMaskPair> pairs;
    pairs.push_back({imageA, maskA});
    pairs.push_back({imageB, maskB});

    EXPECT_NO_THROW(Postprocessor::addFeathering(pairs, 0, 1.0f));
}
