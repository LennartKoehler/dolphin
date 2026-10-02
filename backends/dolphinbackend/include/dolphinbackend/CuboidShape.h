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

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <array>
#include <bit>
#include <cassert>
#include <vector>
#include <algorithm>

struct CuboidShape{
    size_t width;
    size_t height;
    size_t depth;

    CuboidShape() = default;
    CuboidShape(size_t width, size_t height, size_t depth)
        : width(width),
        height(height),
        depth(depth){
        }
    CuboidShape(const std::array<size_t, 3>& dimensions)
        : width(dimensions[0]),
        height(dimensions[1]),
        depth(dimensions[2]){}

    std::array<size_t, 3> getArray() const {
        return std::array<size_t, 3>{width, height, depth};
    }

    size_t& at(size_t dimension) {
        assert(dimension < 3);
        return dimension == 0 ? width : dimension == 1 ? height : depth;
    }
    size_t at(size_t dimension) const {
        assert(dimension < 3);
        return dimension == 0 ? width : dimension == 1 ? height : depth;
    }

    template <typename F>
    CuboidShape& transform(F&& f) {
        f(width); f(height); f(depth);
        return *this;
    }
    template <typename F>
    CuboidShape& transformWith(const CuboidShape& other, F&& f) {
        f(width, other.width); f(height, other.height); f(depth, other.depth);
        return *this;
    }

    size_t getVolume() const {
        return width * height * depth;
    }


    std::string print() const{
        return std::to_string(width) + " x " + std::to_string(height) + " x " + std::to_string(depth);
    }


    inline void clamp(const CuboidShape& maxSize) {
        setMax(maxSize);
    }

    inline size_t getNumberSubcubes(CuboidShape other) const {
        other.setMin(CuboidShape{1, 1, 1});
        CuboidShape temp = this->ceilingDivide(other);
        return temp.getVolume();

    }

    inline void toNextPowerOfTwo(){
        assert(*this > CuboidShape(0,0,0));
        transform([](size_t& d){ d = std::bit_ceil(static_cast<uint32_t>(d)); });
    }



    inline void setMax(const CuboidShape& other){
        transformWith(other, [](size_t& a, size_t b){ a = std::min(a, b); });
    }

    inline void setMin(const CuboidShape& other){
        transformWith(other, [](size_t& a, size_t b){ a = std::max(a, b); });
    }

    inline bool operator==(const CuboidShape& other) const {
        return (this->width == other.width && this->height == other.height && this->depth == other.depth);
    }
    inline bool operator!=(const CuboidShape& other) const {
        return (this->width != other.width || this->height != other.height || this->depth != other.depth);
    }
    inline CuboidShape operator-(const CuboidShape& other) const {
        return CuboidShape(this->width - other.width,
            this->height - other.height,
            this->depth - other.depth);
    }
    inline CuboidShape operator+(const CuboidShape& other) const {
        return CuboidShape(this->width + other.width,
            this->height + other.height,
            this->depth + other.depth);
    }

    inline CuboidShape ceilingDivide(const CuboidShape& other) const {
        return CuboidShape(
            (this->width + other.width - 1) / other.width,
            (this->height + other.height - 1) / other.height,
            (this->depth + other.depth - 1) / other.depth
        );
    }

    inline CuboidShape operator/(const CuboidShape& other) const {
        return CuboidShape(
            this->width/other.width,
            this->height/other.height,
            this->depth/other.depth
        );
    }
    inline CuboidShape& operator+=(const CuboidShape& other) {
        width += static_cast<int64_t>(other.width);
        height += static_cast<int64_t>(other.height);
        depth += static_cast<int64_t>(other.depth);
        return *this;
    }

    inline CuboidShape operator/(const size_t value) const {
        return CuboidShape(this->width/value, this->height/value, this->depth/value);
    }
    inline CuboidShape operator*(const size_t value) const {
        return CuboidShape(this->width*value, this->height*value, this->depth*value);
    }
    inline CuboidShape operator+(const size_t value) const {
        return CuboidShape(this->width+value, this->height+value, this->depth+value);
    }
    inline bool operator>(const CuboidShape& other) const {
        if (this->width > other.width && this->height > other.height && this->depth > other.depth) return true;
        else return false;
    }
    inline bool operator>=(const CuboidShape& other) const {
        if (this->width >= other.width && this->height >= other.height && this->depth >= other.depth) return true;
        else return false;
    }
    inline bool operator<(size_t size) const {
        return (this->width < size || this->height < size || this->depth < size);
    }
    inline bool operator>(size_t size) const {
        return (this->width > size || this->height > size || this->depth > size);
    }
    inline bool operator<(const CuboidShape& other) const {
        return (this->width < other.width || this->height < other.height || this->depth < other.depth);
    }

};

struct CuboidPosition {
    int64_t width;
    int64_t height;
    int64_t depth;

    CuboidPosition() = default;
    CuboidPosition(int64_t width, int64_t height, int64_t depth)
        : width(width), height(height), depth(depth) {}
    CuboidPosition(const CuboidShape& shape)
        : width(static_cast<int64_t>(shape.width)),
          height(static_cast<int64_t>(shape.height)),
          depth(static_cast<int64_t>(shape.depth)) {}

    int64_t& at(size_t dimension) {
        assert(dimension < 3);
        return dimension == 0 ? width : dimension == 1 ? height : depth;
    }
    int64_t at(size_t dimension) const {
        assert(dimension < 3);
        return dimension == 0 ? width : dimension == 1 ? height : depth;
    }

    CuboidPosition operator-(const CuboidPosition& other) const {
        return CuboidPosition(width - other.width, height - other.height, depth - other.depth);
    }
    CuboidPosition operator-(const CuboidShape& other) const {
        return CuboidPosition(width - static_cast<int64_t>(other.width),
                              height - static_cast<int64_t>(other.height),
                              depth - static_cast<int64_t>(other.depth));
    }
    CuboidPosition operator+(const CuboidShape& other) const {
        return CuboidPosition(width + static_cast<int64_t>(other.width),
                              height + static_cast<int64_t>(other.height),
                              depth + static_cast<int64_t>(other.depth));
    }
    CuboidPosition operator+(const CuboidPosition& other) const {
        return CuboidPosition(width + other.width, height + other.height, depth + other.depth);
    }
    CuboidPosition& operator-=(const CuboidShape& other) {
        width -= static_cast<int64_t>(other.width);
        height -= static_cast<int64_t>(other.height);
        depth -= static_cast<int64_t>(other.depth);
        return *this;
    }
    CuboidPosition& operator+=(const CuboidShape& other) {
        width += static_cast<int64_t>(other.width);
        height += static_cast<int64_t>(other.height);
        depth += static_cast<int64_t>(other.depth);
        return *this;
    }
    bool operator==(const CuboidPosition& other) const {
        return width == other.width && height == other.height && depth == other.depth;
    }
    bool operator!=(const CuboidPosition& other) const {
        return !(*this == other);
    }
    bool operator>=(const CuboidShape& other) const {
        return width >= static_cast<int64_t>(other.width) &&
               height >= static_cast<int64_t>(other.height) &&
               depth >= static_cast<int64_t>(other.depth);
    }
    bool operator<(const CuboidShape& other) const {
        return width < static_cast<int64_t>(other.width) ||
               height < static_cast<int64_t>(other.height) ||
               depth < static_cast<int64_t>(other.depth);
    }
    bool operator>(const CuboidShape& other) const {
        return width > static_cast<int64_t>(other.width) ||
               height > static_cast<int64_t>(other.height) ||
               depth > static_cast<int64_t>(other.depth);
    }
    bool operator<=(const CuboidShape& other) const {
        return width <= static_cast<int64_t>(other.width) &&
               height <= static_cast<int64_t>(other.height) &&
               depth <= static_cast<int64_t>(other.depth);
    }

    CuboidShape toShape() const {
        return CuboidShape(static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(depth));
    }

    std::string print() const {
        return std::to_string(width) + " x " + std::to_string(height) + " x " + std::to_string(depth);
    }
};

inline CuboidShape operator-(const CuboidShape& s, const CuboidPosition& p) {
    return CuboidShape(static_cast<size_t>(static_cast<int64_t>(s.width) - p.width),
                       static_cast<size_t>(static_cast<int64_t>(s.height) - p.height),
                       static_cast<size_t>(static_cast<int64_t>(s.depth) - p.depth));
}

inline CuboidShape getLargestShape(const std::vector<CuboidShape>& psfSizes) {
    CuboidShape maxPsfShape{0, 0, 0};
    for (const auto& psf : psfSizes) {
        maxPsfShape.transformWith(psf, [](size_t& a, size_t b){ a = std::max(a, b); });
    }
    return maxPsfShape;
}
