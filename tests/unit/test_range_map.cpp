#include <gtest/gtest.h>
#include "dolphin_image/Types/Range.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphinbackend/CuboidShape.h"
#include <string>

TEST(RangeTest, ConstructorSingle) {
    Range<std::string> r(0, 5, "hello");
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 5);
    ASSERT_EQ(r.values.size(), 1u);
    EXPECT_EQ(r.values[0], "hello");
}

TEST(RangeTest, ConstructorVector) {
    std::vector<std::string> vals = {"a", "b"};
    Range<std::string> r(0, 10, vals);
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 10);
    EXPECT_EQ(r.values.size(), 2u);
}

TEST(RangeTest, Contains) {
    Range<std::string> r(0, 5, "val");
    EXPECT_TRUE(r.contains(0));
    EXPECT_TRUE(r.contains(4));
    EXPECT_FALSE(r.contains(5));
    EXPECT_FALSE(r.contains(-1));
}

TEST(RangeTest, ContainsSingleIndex) {
    Range<std::string> r(3, 3, "val");
    EXPECT_TRUE(r.contains(3));
    EXPECT_FALSE(r.contains(2));
    EXPECT_FALSE(r.contains(4));
}

TEST(RangeTest, AddValues) {
    Range<std::string> r(0, 5, "a");
    std::vector<std::string> more = {"b", "c"};
    r.addValues(more);
    ASSERT_EQ(r.values.size(), 3u);
    EXPECT_EQ(r.values[0], "a");
    EXPECT_EQ(r.values[1], "b");
    EXPECT_EQ(r.values[2], "c");
}

TEST(RangeMapTest, AddRange) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "psf_a");
    auto results = map.get(2);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].values[0], "psf_a");
}

TEST(RangeMapTest, MultipleRanges) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "psf_a");
    map.addRange(5, 10, "psf_b");
    EXPECT_EQ(map.get(2).size(), 1u);
    EXPECT_EQ(map.get(7).size(), 1u);
    EXPECT_EQ(map.get(5).size(), 1u);
}

TEST(RangeMapTest, OverlappingRanges) {
    RangeMap<std::string> map;
    map.addRange(0, 10, "psf_a");
    map.addRange(5, 15, "psf_b");
    auto results = map.get(7);
    EXPECT_EQ(results.size(), 2u);
}

TEST(RangeMapTest, SameRangeAddsValue) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "psf_a");
    map.addRange(0, 5, "psf_b");
    auto results = map.get(2);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].values.size(), 2u);
}

TEST(RangeMapTest, Empty) {
    RangeMap<std::string> map;
    EXPECT_TRUE(map.empty());
    map.addRange(0, 5, "val");
    EXPECT_FALSE(map.empty());
}

TEST(RangeMapTest, GetNotFound) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "val");
    auto results = map.get(100);
    EXPECT_TRUE(results.empty());
}

TEST(RangeMapTest, GetPointers) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "val");
    auto ptrs = map.getPointers(2);
    ASSERT_EQ(ptrs.size(), 1u);
    EXPECT_EQ(*ptrs[0], "val");
}

TEST(RangeMapTest, Clear) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "val");
    map.clear();
    EXPECT_TRUE(map.empty());
}

TEST(RangeMapTest, LoadFromStringSingle) {
    RangeMap<std::string> map;
    map.loadFromString("0[ID1234]");
    auto results = map.get(0);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].values[0], "ID1234");
}

TEST(RangeMapTest, LoadFromStringRange) {
    RangeMap<std::string> map;
    map.loadFromString("0:5[ID1234]");
    EXPECT_EQ(map.get(0).size(), 1u);
    EXPECT_EQ(map.get(4).size(), 1u);
    EXPECT_EQ(map.get(5).size(), 0u);
}

TEST(RangeMapTest, LoadFromStringMultiple) {
    RangeMap<std::string> map;
    map.loadFromString("0[ID1234], 1[constant_hyperstack_1]");
    EXPECT_EQ(map.get(0).size(), 1u);
    EXPECT_EQ(map.get(0)[0].values[0], "ID1234");
    EXPECT_EQ(map.get(1).size(), 1u);
    EXPECT_EQ(map.get(1)[0].values[0], "constant_hyperstack_1");
}

TEST(RangeMapTest, LoadFromStringMultipleValues) {
    RangeMap<std::string> map;
    map.loadFromString("0:5[ID1234,ID5678]");
    auto results = map.get(2);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].values.size(), 2u);
    EXPECT_EQ(results[0].values[0], "ID1234");
    EXPECT_EQ(results[0].values[1], "ID5678");
}

TEST(RangeMapTest, LoadFromStringInvalid) {
    RangeMap<std::string> map;
    EXPECT_NO_THROW(map.loadFromString("invalid"));
}

TEST(RangeMapTest, IteratorBasic) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "a");
    map.addRange(5, 10, "b");
    int count = 0;
    for (const auto& range : map) {
        count++;
    }
    EXPECT_EQ(count, 2);
}

TEST(RangeMapTest, CopyConstructor) {
    RangeMap<std::string> map;
    map.addRange(0, 5, "val");
    RangeMap<std::string> copy(map);
    EXPECT_EQ(copy.get(2).size(), 1u);
}

class BoxCoordRangeTest : public ::testing::Test {
protected:
    BoxCoord inner, outer;
    void SetUp() override {
        outer.position = CuboidShape(0, 0, 0);
        outer.dimensions = CuboidShape(10, 10, 10);
        inner.position = CuboidShape(2, 2, 2);
        inner.dimensions = CuboidShape(4, 4, 4);
    }
};

TEST_F(BoxCoordRangeTest, IsWithinTrue) {
    EXPECT_TRUE(inner.isWithin(outer));
}

TEST_F(BoxCoordRangeTest, IsWithinFalse) {
    BoxCoord outside;
    outside.position = CuboidShape(8, 8, 8);
    outside.dimensions = CuboidShape(5, 5, 5);
    EXPECT_FALSE(outside.isWithin(outer));
}

TEST_F(BoxCoordRangeTest, IsWithinEqual) {
    EXPECT_TRUE(outer.isWithin(outer));
}

TEST_F(BoxCoordRangeTest, Print) {
    BoxCoord bc;
    bc.position = CuboidShape(1, 2, 3);
    bc.dimensions = CuboidShape(4, 5, 6);
    std::string s = bc.print();
    EXPECT_FALSE(s.empty());
}

TEST_F(BoxCoordRangeTest, CropToNoCrop) {
    BoxCoord box;
    box.position = CuboidShape(2, 2, 2);
    box.dimensions = CuboidShape(4, 4, 4);
    Padding padding = box.cropTo(outer);
    EXPECT_EQ(padding.before.width, 0);
    EXPECT_EQ(padding.before.height, 0);
    EXPECT_EQ(padding.before.depth, 0);
}

TEST_F(BoxCoordRangeTest, CropToPartialOverlap) {
    BoxCoord box;
    box.position = CuboidShape(8, 8, 8);
    box.dimensions = CuboidShape(4, 4, 4);
    Padding padding = box.cropTo(outer);
    EXPECT_EQ(box.dimensions.width, 2);
    EXPECT_EQ(padding.before.width, 0);
    EXPECT_EQ(padding.after.width, 2);
}

TEST(PaddingTest, GetTotalPadding) {
    Padding p;
    p.before = CuboidShape(1, 2, 3);
    p.after = CuboidShape(4, 5, 6);
    CuboidShape total = p.getTotalPadding();
    EXPECT_EQ(total.width, 5);
    EXPECT_EQ(total.height, 7);
    EXPECT_EQ(total.depth, 9);
}

TEST(PaddingTest, GetPaddingWidthTotal) {
    Padding p;
    p.before = CuboidShape(2, 0, 0);
    p.after = CuboidShape(3, 0, 0);
    EXPECT_EQ(p.getPaddingWidthTotal(), 5);
}

TEST(PaddingTest, GetPaddingHeightTotal) {
    Padding p;
    p.before = CuboidShape(0, 3, 0);
    p.after = CuboidShape(0, 4, 0);
    EXPECT_EQ(p.getPaddingHeightTotal(), 7);
}

TEST(BoxCoordWithPaddingTest, GetPaddedShape) {
    BoxCoordWithPadding bcp;
    bcp.box.position = CuboidShape(0, 0, 0);
    bcp.box.dimensions = CuboidShape(10, 10, 10);
    bcp.padding.before = CuboidShape(2, 2, 2);
    bcp.padding.after = CuboidShape(3, 3, 3);
    CuboidShape padded = bcp.getPaddedShape();
    EXPECT_EQ(padded.width, 15);
    EXPECT_EQ(padded.height, 15);
    EXPECT_EQ(padded.depth, 15);
}

TEST(BoxCoordWithPaddingTest, GetBox) {
    BoxCoordWithPadding bcp;
    bcp.box.position = CuboidShape(5, 5, 5);
    bcp.box.dimensions = CuboidShape(10, 10, 10);
    bcp.padding.before = CuboidShape(2, 2, 2);
    bcp.padding.after = CuboidShape(3, 3, 3);
    BoxCoord expanded = bcp.getBox();
    EXPECT_EQ(expanded.position.width, 3);
    EXPECT_EQ(expanded.dimensions.width, 15);
}

TEST(BoxCoordWithPaddingTest, IsWithin) {
    BoxCoordWithPadding inner, outer;
    outer.box.position = CuboidShape(0, 0, 0);
    outer.box.dimensions = CuboidShape(20, 20, 20);
    outer.padding.before = CuboidShape(0, 0, 0);
    outer.padding.after = CuboidShape(0, 0, 0);
    inner.box.position = CuboidShape(5, 5, 5);
    inner.box.dimensions = CuboidShape(5, 5, 5);
    inner.padding.before = CuboidShape(0, 0, 0);
    inner.padding.after = CuboidShape(0, 0, 0);
    EXPECT_TRUE(inner.isWithin(outer));
}
