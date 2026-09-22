#include <gtest/gtest.h>
#include "dolphinbackend/CuboidShape.h"
#include <vector>

TEST(CuboidShapeTest, ParameterizedConstructorZeros) {
    CuboidShape s(0, 0, 0);
    EXPECT_EQ(s.width, 0);
    EXPECT_EQ(s.height, 0);
    EXPECT_EQ(s.depth, 0);
}

TEST(CuboidShapeTest, ParameterizedConstructor) {
    CuboidShape s(10, 20, 30);
    EXPECT_EQ(s.width, 10);
    EXPECT_EQ(s.height, 20);
    EXPECT_EQ(s.depth, 30);
}

TEST(CuboidShapeTest, ArrayConstructor) {
    CuboidShape s(std::array<size_t, 3>{5, 10, 15});
    EXPECT_EQ(s.width, 5);
    EXPECT_EQ(s.height, 10);
    EXPECT_EQ(s.depth, 15);
}

TEST(CuboidShapeTest, GetArray) {
    CuboidShape s(3, 6, 9);
    auto arr = s.getArray();
    EXPECT_EQ(arr[0], 3);
    EXPECT_EQ(arr[1], 6);
    EXPECT_EQ(arr[2], 9);
}

TEST(CuboidShapeTest, GetVolume) {
    EXPECT_EQ(CuboidShape(2, 3, 4).getVolume(), 24);
    EXPECT_EQ(CuboidShape(1, 1, 1).getVolume(), 1);
    EXPECT_EQ(CuboidShape(0, 5, 5).getVolume(), 0);
}

TEST(CuboidShapeTest, Print) {
    CuboidShape s(10, 20, 30);
    EXPECT_EQ(s.print(), "10 x 20 x 30");
}

TEST(CuboidShapeTest, EqualityOperators) {
    CuboidShape a(1, 2, 3);
    CuboidShape b(1, 2, 3);
    CuboidShape c(3, 2, 1);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a != b);
}

TEST(CuboidShapeTest, Addition) {
    CuboidShape a(1, 2, 3);
    CuboidShape b(4, 5, 6);
    CuboidShape c = a + b;
    EXPECT_EQ(c.width, 5);
    EXPECT_EQ(c.height, 7);
    EXPECT_EQ(c.depth, 9);
}

TEST(CuboidShapeTest, Subtraction) {
    CuboidShape a(10, 20, 30);
    CuboidShape b(1, 2, 3);
    CuboidShape c = a - b;
    EXPECT_EQ(c.width, 9);
    EXPECT_EQ(c.height, 18);
    EXPECT_EQ(c.depth, 27);
}

TEST(CuboidShapeTest, DivisionByShape) {
    CuboidShape a(10, 20, 30);
    CuboidShape b(2, 4, 5);
    CuboidShape c = a / b;
    EXPECT_EQ(c.width, 5);
    EXPECT_EQ(c.height, 5);
    EXPECT_EQ(c.depth, 6);
}

TEST(CuboidShapeTest, DivisionByInt) {
    CuboidShape a(10, 20, 30);
    CuboidShape c = a / 5;
    EXPECT_EQ(c.width, 2);
    EXPECT_EQ(c.height, 4);
    EXPECT_EQ(c.depth, 6);
}

TEST(CuboidShapeTest, MultiplicationByInt) {
    CuboidShape a(2, 3, 4);
    CuboidShape c = a * size_t{3};
    EXPECT_EQ(c.width, 6);
    EXPECT_EQ(c.height, 9);
    EXPECT_EQ(c.depth, 12);
}

TEST(CuboidShapeTest, MultiplicationByDouble) {
    CuboidShape a(2, 3, 4);
    CuboidShape c = a * 2.5;
    EXPECT_EQ(c.width, 5);
    EXPECT_EQ(c.height, 7);
    EXPECT_EQ(c.depth, 10);
}

TEST(CuboidShapeTest, AdditionByInt) {
    CuboidShape a(2, 3, 4);
    CuboidShape c = a + 5;
    EXPECT_EQ(c.width, 7);
    EXPECT_EQ(c.height, 8);
    EXPECT_EQ(c.depth, 9);
}

TEST(CuboidShapeTest, CeilingDivide) {
    CuboidShape a(10, 20, 30);
    CuboidShape b(3, 7, 4);
    CuboidShape c = a.ceilingDivide(b);
    EXPECT_EQ(c.width, 4);
    EXPECT_EQ(c.height, 3);
    EXPECT_EQ(c.depth, 8);
}

TEST(CuboidShapeTest, CeilingDivideExact) {
    CuboidShape a(10, 20, 30);
    CuboidShape b(2, 4, 5);
    CuboidShape c = a.ceilingDivide(b);
    EXPECT_EQ(c.width, 5);
    EXPECT_EQ(c.height, 5);
    EXPECT_EQ(c.depth, 6);
}

TEST(CuboidShapeTest, ToNextPowerOfTwo) {
    CuboidShape s(3, 5, 7);
    s.toNextPowerOfTwo();
    EXPECT_EQ(s.width, 4);
    EXPECT_EQ(s.height, 8);
    EXPECT_EQ(s.depth, 8);
}

TEST(CuboidShapeTest, ToNextPowerOfTwoAlreadyPower) {
    CuboidShape s(4, 8, 16);
    s.toNextPowerOfTwo();
    EXPECT_EQ(s.width, 4);
    EXPECT_EQ(s.height, 8);
    EXPECT_EQ(s.depth, 16);
}

TEST(CuboidShapeTest, Clamp) {
    CuboidShape s(10, 20, 30);
    s.clamp(CuboidShape(5, 30, 15));
    EXPECT_EQ(s.width, 5);
    EXPECT_EQ(s.height, 20);
    EXPECT_EQ(s.depth, 15);
}

TEST(CuboidShapeTest, SetMax) {
    CuboidShape s(10, 20, 30);
    s.setMax(CuboidShape(5, 50, 15));
    EXPECT_EQ(s.width, 5);
    EXPECT_EQ(s.height, 20);
    EXPECT_EQ(s.depth, 15);
}

TEST(CuboidShapeTest, SetMin) {
    CuboidShape s(10, 20, 30);
    s.setMin(CuboidShape(5, 50, 15));
    EXPECT_EQ(s.width, 10);
    EXPECT_EQ(s.height, 50);
    EXPECT_EQ(s.depth, 30);
}

TEST(CuboidShapeTest, GreaterThan) {
    CuboidShape a(10, 20, 30);
    CuboidShape b(5, 10, 15);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
}

TEST(CuboidShapeTest, GreaterThanOrEqual) {
    CuboidShape a(10, 20, 30);
    CuboidShape b(10, 20, 30);
    EXPECT_TRUE(a >= b);
}

TEST(CuboidShapeTest, LessThanShape) {
    CuboidShape a(5, 20, 30);
    CuboidShape b(10, 10, 10);
    EXPECT_TRUE(a < b);
}

TEST(CuboidShapeTest, GetNumberSubcubes) {
    CuboidShape image(100, 100, 100);
    CuboidShape cube(25, 25, 25);
    EXPECT_EQ(image.getNumberSubcubes(cube), 64);
}

TEST(CuboidShapeTest, GetNumberSubcubesNonDivisible) {
    CuboidShape image(100, 100, 100);
    CuboidShape cube(30, 30, 30);
    EXPECT_EQ(image.getNumberSubcubes(cube), 64);
}

TEST(CuboidShapeTest, GetLargestShape) {
    std::vector<CuboidShape> shapes = {
        {10, 20, 30},
        {15, 10, 25},
        {5, 25, 15}
    };
    CuboidShape largest = getLargestShape(shapes);
    EXPECT_EQ(largest.width, 15);
    EXPECT_EQ(largest.height, 25);
    EXPECT_EQ(largest.depth, 30);
}

TEST(CuboidShapeTest, GetLargestShapeEmpty) {
    std::vector<CuboidShape> shapes;
    CuboidShape largest = getLargestShape(shapes);
    EXPECT_EQ(largest.width, 0);
    EXPECT_EQ(largest.height, 0);
    EXPECT_EQ(largest.depth, 0);
}

TEST(CuboidShapeTest, GetReference) {
    CuboidShape s(10, 20, 30);
    auto refs = s.getReference();
    EXPECT_EQ(*refs[0], 10);
    EXPECT_EQ(*refs[1], 20);
    EXPECT_EQ(*refs[2], 30);
    *refs[0] = 99;
    EXPECT_EQ(s.width, 99);
}
