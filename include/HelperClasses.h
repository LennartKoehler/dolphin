#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <opencv2/core/mat.hpp>

// Forward declaration
class Image3D;

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
    void cropTo(const RectangleShape& other){
        this->width = std::min(this->width, other.width);
        this->height = std::min(this->height, other.height);
        this->depth = std::min(this->depth, other.depth);
        updateVolume();
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
    Image3D image;
    Padding padding;
};

struct BoxCoord {
    RectangleShape position;
    RectangleShape dimensions;
    bool isWithin(const BoxCoord& other) const {
        // Check if this box is completely within the other box
        return (position.width >= other.position.width &&
                position.height >= other.position.height &&
                position.depth >= other.position.depth &&
                position.width + dimensions.width <= other.position.width + other.dimensions.width &&
                position.height + dimensions.height <= other.position.height + other.dimensions.height &&
                position.depth + dimensions.depth <= other.position.depth + other.dimensions.depth);
    }
    Padding cropTo(const BoxCoord& other) {
        // Store original values to calculate what was cropped
        RectangleShape originalPosition = position;
        RectangleShape originalDimensions = dimensions;
        
        // Adjust position to be within the other box
        position.width = std::max(position.width, other.position.width);
        position.height = std::max(position.height, other.position.height);
        position.depth = std::max(position.depth, other.position.depth);
        
        // Calculate the maximum allowed dimensions
        int maxWidth = std::max(0, other.position.width + other.dimensions.width - position.width);
        int maxHeight = std::max(0, other.position.height + other.dimensions.height - position.height);
        int maxDepth = std::max(0, other.position.depth + other.dimensions.depth - position.depth);
        
        // Crop dimensions to fit within the other box
        dimensions.width = std::min(dimensions.width, maxWidth);
        dimensions.height = std::min(dimensions.height, maxHeight);
        dimensions.depth = std::min(dimensions.depth, maxDepth);
        dimensions.updateVolume();
        
        // Calculate cropped amounts
        Padding croppedPadding;
        croppedPadding.before.width = position.width - originalPosition.width;
        croppedPadding.before.height = position.height - originalPosition.height;
        croppedPadding.before.depth = position.depth - originalPosition.depth;
        
        croppedPadding.after.width = originalDimensions.width - dimensions.width - croppedPadding.before.width;
        croppedPadding.after.height = originalDimensions.height - dimensions.height - croppedPadding.before.height;
        croppedPadding.after.depth = originalDimensions.depth - dimensions.depth - croppedPadding.before.depth;
        
        return croppedPadding;
    }
};

struct BoxCoordWithPadding {
    BoxCoord box;
    Padding padding;
    bool isWithin(const BoxCoordWithPadding& other) const {
        return (box.isWithin(other.box) && other.padding.before >= padding.before && other.padding.after >= padding.after);
    }
};

template<typename entryType>
struct BoxEntryPair {
    BoxCoord box;
    entryType entry;
    
    BoxEntryPair(BoxCoord b, entryType p) 
        : box(b), entry(std::move(p)) {}
};

