#include <gtest/gtest.h>
#include "dolphinbackend/CuboidShape.h"
#include "dolphin/Logging.h"
#include <chrono>

class CubeShapeIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(CubeShapeIntegrationTest, PrintFormatting) {
    CuboidShape s(128, 128, 64);
    EXPECT_EQ(s.print(), "128 x 128 x 64");
}

TEST_F(CubeShapeIntegrationTest, VolumeCalculation) {
    CuboidShape s(128, 128, 64);
    EXPECT_EQ(s.getVolume(), 1048576);
}

TEST_F(CubeShapeIntegrationTest, NextPowerOfTwo) {
    CuboidShape s(100, 128, 64);
    s.toNextPowerOfTwo();
    EXPECT_EQ(s.width, 128);
    EXPECT_EQ(s.height, 128);
    EXPECT_EQ(s.depth, 64);
}

TEST_F(CubeShapeIntegrationTest, CeilingDivideForCubes) {
    CuboidShape image(256, 256, 64);
    CuboidShape cube(64, 64, 32);
    CuboidShape numCubes = image.ceilingDivide(cube);
    EXPECT_EQ(numCubes.width, 4);
    EXPECT_EQ(numCubes.height, 4);
    EXPECT_EQ(numCubes.depth, 2);
    EXPECT_EQ(numCubes.getVolume(), 32);
}
