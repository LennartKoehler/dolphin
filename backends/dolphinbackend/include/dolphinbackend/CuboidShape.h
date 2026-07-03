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

    std::array<size_t*, 3> getReference() {
        return std::array<size_t*, 3>{&width, &height, &depth};
    }

    size_t getVolume() const {
        return width * height * depth;
    }


    std::string print() const{
        return std::to_string(width) + " x " + std::to_string(height) + " x " + std::to_string(depth);
    }

    inline void clamp(const CuboidShape& maxSize) {
        width  = width  < maxSize.width  ? width  : maxSize.width;
        height = height < maxSize.height ? height : maxSize.height;
        depth  = depth  < maxSize.depth  ? depth  : maxSize.depth;
    }

    inline size_t getNumberSubcubes(CuboidShape other) const {
        other.setMin(CuboidShape{1, 1, 1});
        CuboidShape temp = this->ceilingDivide(other);
        return temp.getVolume();

    }

    inline void toNextPowerOfTwo(){
        assert(*this > CuboidShape(0,0,0));
        width = std::bit_ceil(static_cast<uint>(width));
        height = std::bit_ceil(static_cast<uint>(height));
        depth = std::bit_ceil(static_cast<uint>(depth));
    }



    inline void setMax(const CuboidShape& other){
        this->width = this->width < other.width ? this->width : other.width;
        this->height = this->height < other.height ? this->height : other.height;
        this->depth = this->depth < other.depth ? this->depth : other.depth;
    }

    inline void setMin(const CuboidShape& other){
        this->width = this->width > other.width ? this->width : other.width;
        this->height = this->height > other.height ? this->height : other.height;
        this->depth = this->depth > other.depth ? this->depth : other.depth;
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

    inline CuboidShape operator/(const size_t value) const {
        return CuboidShape(this->width/value, this->height/value, this->depth/value);
    }
    inline CuboidShape operator*(const size_t value) const {
        return CuboidShape(this->width*value, this->height*value, this->depth*value);
    }
    inline CuboidShape operator*(const double value) const {
        return CuboidShape(static_cast<size_t>(this->width*value), static_cast<size_t>(this->height*value), static_cast<size_t>(this->depth*value));
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
        maxPsfShape.width = std::max(maxPsfShape.width, psf.width);
        maxPsfShape.height = std::max(maxPsfShape.height, psf.height);
        maxPsfShape.depth = std::max(maxPsfShape.depth, psf.depth);
    }
    return maxPsfShape;
}
