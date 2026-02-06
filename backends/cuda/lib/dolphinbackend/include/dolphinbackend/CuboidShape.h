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

#include <string>
#include <array>

struct CuboidShape{
    int width;
    int height;
    int depth;

    CuboidShape() = default;
    CuboidShape(int width, int height, int depth)
        : width(width),
        height(height),
        depth(depth){
        }
    CuboidShape(const std::array<int, 3>& dimensions)
        : width(dimensions[0]),
        height(dimensions[1]),
        depth(dimensions[2]){}
    
    std::array<int, 3> getArray() const {
        return std::array<int, 3>{width, height, depth};
    }

    int getVolume() const {
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
    // std::array<int*, 3> getDimensionsAscending()
    // {
    //     // Create an array of pointers to members
    //     std::array<int*, 3> dims = { &width, &height, &depth };

    //     // Sort pointers based on the values they point to
    //     std::sort(dims.begin(), dims.end(),
    //         [](const int* a, const int* b) {
    //             return *a < *b;  // ascending order
    //         });

    //     return dims;
    // }

    inline int getNumberSubcubes(const CuboidShape& other) const {
        CuboidShape temp = this->operator/(other);
        temp.setMin(CuboidShape{1, 1, 1});
        return temp.getVolume();

    }

    inline void cropTo(const CuboidShape& other) {
        width  = width  < other.width  ? width  : other.width;
        height = height < other.height ? height : other.height;
        depth  = depth  < other.depth  ? depth  : other.depth;
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

    inline CuboidShape operator/(const CuboidShape& other) const {
        return CuboidShape(
            this->width/other.width,
            this->height/other.height,
            this->depth/other.depth
        );
    }

    inline CuboidShape operator/(const int value) const {
        return CuboidShape(this->width/value, this->height/value, this->depth/value);
    }
    inline CuboidShape operator*(const int value) const {
        return CuboidShape(this->width*value, this->height*value, this->depth*value);
    }
    inline CuboidShape operator*(const double value) const {
        return CuboidShape(this->width*value, this->height*value, this->depth*value);
    }
    inline CuboidShape operator+(const int value) const {
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
    inline bool operator<(const CuboidShape& other) const {
        if (this->width != other.width) return this->width < other.width;
        if (this->height != other.height) return this->height < other.height;
        return this->depth < other.depth;
    }
};

