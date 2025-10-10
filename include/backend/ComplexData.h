#pragma once
#include <array>
#include <algorithm>

class IBackendMemoryManager;

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
    inline bool operator!=(const RectangleShape& other){
        return (this->height != other.height || this->width != other.width || this->depth != other.depth || this->volume != other.volume);
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
};

typedef double complex[2];


class ComplexData{
public:
    complex* data;
    RectangleShape size;
    IBackendMemoryManager* backend;

    // Take ownership of pre-allocated memory
    ComplexData(IBackendMemoryManager* b, complex* data, RectangleShape size) : backend(b), size(size), data(data){};
    ComplexData(const ComplexData& other) = delete;
    ComplexData& operator=(const ComplexData& other) = delete;


    ComplexData(ComplexData&& other) noexcept
     : data(other.data), backend(other.backend){
        other.data = nullptr;
     };
};