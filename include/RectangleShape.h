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
            volume = width * height * depth;
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
    inline RectangleShape operator+(const int value) const {
        return RectangleShape(this->width+value, this->height+value, this->depth+value);
    }
    inline bool operator>(const RectangleShape& other) const {
        if (this->width > other.width && this->height > other.height && this->depth > other.depth) return true;
        else return false;
    }

};