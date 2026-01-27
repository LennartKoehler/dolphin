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

    inline void clamp(const RectangleShape& maxSize) {
        width  = width  < maxSize.width  ? width  : maxSize.width;
        height = height < maxSize.height ? height : maxSize.height;
        depth  = depth  < maxSize.depth  ? depth  : maxSize.depth;
        updateVolume();
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
    inline void cropTo(const RectangleShape& other) {
        width  = width  < other.width  ? width  : other.width;
        height = height < other.height ? height : other.height;
        depth  = depth  < other.depth  ? depth  : other.depth;
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

