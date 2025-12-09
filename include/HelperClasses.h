#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <opencv2/core/mat.hpp>

struct RectangleShape{
    int width;
    int height;
    int depth;
    int volume;

    RectangleShape() = default;
    RectangleShape(int width, int height, int depth)
        : width(width),
        height(height),
        depth(depth){
           updateVolume(); 
        }
    inline void updateVolume(){
        volume = width * height * depth;
    }
    inline void clamp(const RectangleShape& maxSize){
        this->width = std::min(this->width, maxSize.width);
        this->height = std::min(this->height, maxSize.height);
        this->depth = std::min(this->depth, maxSize.depth);
        this->volume = width * height * depth;
    }
    std::array<int*, 3> getDimensionsAscending()
    {
        // Create an array of pointers to members
        std::array<int*, 3> dims = { &width, &height, &depth };

        // Sort pointers based on the values they point to
        std::sort(dims.begin(), dims.end(),
            [](const int* a, const int* b) {
                return *a < *b;  // ascending order
            });

        return dims;
    }
    inline bool operator==(const RectangleShape& other) const {
        return (this->width == other.width && this->height == other.height && this->depth == other.depth);
    }
    inline bool operator!=(const RectangleShape& other) const {
        return (this->width != other.width || this->height != other.height || this->depth != other.depth);
    }
    inline RectangleShape operator-(const RectangleShape& other) const {
        return RectangleShape(this->width - other.width,
            this->height - other.height,
            this->depth - other.depth);
    }
    inline RectangleShape operator+(const RectangleShape& other) const {
        return RectangleShape(this->width + other.width,
            this->height + other.height,
            this->depth + other.depth);
    }
    inline RectangleShape operator/(const int value) const {
        return RectangleShape(this->width/value, this->height/value, this->depth/value);
    }
    inline RectangleShape operator*(const int value) const {
        return RectangleShape(this->width*value, this->height*value, this->depth*value);
    }
    inline RectangleShape operator*(const double value) const {
        return RectangleShape(this->width*value, this->height*value, this->depth*value);
    }
    inline RectangleShape operator+(const int value) const {
        return RectangleShape(this->width+value, this->height+value, this->depth+value);
    }
    inline bool operator>(const RectangleShape& other) const {
        if (this->width > other.width && this->height > other.height && this->depth > other.depth) return true;
        else return false;
    }
    inline bool operator>=(const RectangleShape& other) const {
        if (this->width >= other.width && this->height >= other.height && this->depth >= other.depth) return true;
        else return false;
    }
    inline bool operator<(const RectangleShape& other) const {
        if (this->width != other.width) return this->width < other.width;
        if (this->height != other.height) return this->height < other.height;
        return this->depth < other.depth;
    }
};

struct Padding{
    RectangleShape before;
    RectangleShape after;
};


struct PaddedImage{
    std::vector<cv::Mat> image;
    Padding padding;
};

struct BoxCoord {
    int x, y, z;
    RectangleShape dimensions;
    bool isWithin(const BoxCoord& other) const {
        // Check if this box is completely within the other box
        return (x >= other.x && 
                y >= other.y && 
                z >= other.z &&
                x + dimensions.width <= other.x + other.dimensions.width &&
                y + dimensions.height <= other.y + other.dimensions.height &&
                z + dimensions.depth <= other.z + other.dimensions.depth);
    }
};

struct BoxCoordWithPadding {
    BoxCoord box;
    Padding padding;
};

template<typename entryType>
struct BoxEntryPair {
    BoxCoord box;
    entryType entry;
    
    BoxEntryPair(BoxCoord b, entryType p) 
        : box(b), entry(std::move(p)) {}
};

