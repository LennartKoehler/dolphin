#include <gtest/gtest.h>
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphinbackend/CuboidShape.h"

TEST(BoxCoordTest, ConstructWithValues) {
    BoxCoord bc;
    bc.position = CuboidShape(0, 0, 0);
    bc.dimensions = CuboidShape(10, 10, 10);
    EXPECT_EQ(bc.position.width, 0);
    EXPECT_EQ(bc.dimensions.width, 10);
}

TEST(BoxCoordTest, IsWithinTrue) {
    BoxCoord outer;
    outer.position = CuboidShape(0, 0, 0);
    outer.dimensions = CuboidShape(10, 10, 10);

    BoxCoord inner;
    inner.position = CuboidShape(2, 2, 2);
    inner.dimensions = CuboidShape(4, 4, 4);

    EXPECT_TRUE(inner.isWithin(outer));
}

TEST(BoxCoordTest, IsWithinAtBoundary) {
    BoxCoord outer;
    outer.position = CuboidShape(0, 0, 0);
    outer.dimensions = CuboidShape(10, 10, 10);

    BoxCoord inner;
    inner.position = CuboidShape(0, 0, 0);
    inner.dimensions = CuboidShape(10, 10, 10);

    EXPECT_TRUE(inner.isWithin(outer));
}

TEST(BoxCoordTest, IsWithinFalse) {
    BoxCoord outer;
    outer.position = CuboidShape(0, 0, 0);
    outer.dimensions = CuboidShape(10, 10, 10);

    BoxCoord outside;
    outside.position = CuboidShape(8, 8, 8);
    outside.dimensions = CuboidShape(5, 5, 5);

    EXPECT_FALSE(outside.isWithin(outer));
}

TEST(BoxCoordTest, Print) {
    BoxCoord bc;
    bc.position = CuboidShape(1, 2, 3);
    bc.dimensions = CuboidShape(4, 5, 6);
    std::string s = bc.print();
    EXPECT_NE(s.find("1"), std::string::npos);
}

TEST(BoxCoordTest, CropToNoOverlap) {
    BoxCoord outer;
    outer.position = CuboidShape(0, 0, 0);
    outer.dimensions = CuboidShape(10, 10, 10);

    BoxCoord box;
    box.position = CuboidShape(20, 20, 20);
    box.dimensions = CuboidShape(5, 5, 5);

    Padding padding = box.cropTo(outer);
    EXPECT_EQ(box.dimensions.width, 0);
}

TEST(PaddingTest, TotalPadding) {
    Padding p;
    p.before = CuboidShape(1, 2, 3);
    p.after = CuboidShape(4, 5, 6);
    EXPECT_EQ(p.getTotalPadding(), CuboidShape(5, 7, 9));
}

TEST(PaddingTest, WidthTotal) {
    Padding p;
    p.before = CuboidShape(2, 0, 0);
    p.after = CuboidShape(3, 0, 0);
    EXPECT_EQ(p.getPaddingWidthTotal(), 5);
}

TEST(BoxCoordWithPaddingTest, PaddedShape) {
    BoxCoordWithPadding bcp;
    bcp.box.position = CuboidShape(0, 0, 0);
    bcp.box.dimensions = CuboidShape(10, 10, 10);
    bcp.padding.before = CuboidShape(2, 2, 2);
    bcp.padding.after = CuboidShape(3, 3, 3);
    EXPECT_EQ(bcp.getPaddedShape(), CuboidShape(15, 15, 15));
}

TEST(BoxCoordWithPaddingTest, GetExpandedBox) {
    BoxCoordWithPadding bcp;
    bcp.box.position = CuboidShape(5, 5, 5);
    bcp.box.dimensions = CuboidShape(10, 10, 10);
    bcp.padding.before = CuboidShape(2, 2, 2);
    bcp.padding.after = CuboidShape(3, 3, 3);
    BoxCoord expanded = bcp.getPaddedBox();
    EXPECT_EQ(expanded.position, CuboidShape(3, 3, 3));
    EXPECT_EQ(expanded.dimensions, CuboidShape(15, 15, 15));
}
