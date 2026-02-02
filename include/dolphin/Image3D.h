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

#include <vector>
#include <list>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <itkExtractImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageDuplicator.h>
#include "dolphin/HelperClasses.h"

using PixelType = float;
constexpr unsigned int Dimension = 3;
using ImageType = itk::Image<PixelType, Dimension>;

// Structure to hold both value and coordinates
struct PixelData {
    float value;
    int x, y, z;
    
    PixelData(float v, int x_coord, int y_coord, int z_coord) 
        : value(v), x(x_coord), y(y_coord), z(z_coord) {}
    
    // Allow implicit conversion to float for backward compatibility
    operator float() const { return value; }
    operator float&() { return value; }
};


class Image3D {
private:

    ImageType::Pointer image;

public:
    Image3D() {
        image = ImageType::New();
    }
    
    Image3D(ImageType::Pointer&& itkImage) {
        this->image = std::move(itkImage);
    }
    
    
    Image3D(const CuboidShape& shape);
    
    Image3D(const Image3D& other);

    Image3D getInRange(float min, float max) const;

    
    Image3D getSubimageCopy(const BoxCoord& coords);
    
    // Accessor methods
    ImageType::Pointer getItkImage() const { return image; }
    void setItkImage(ImageType::Pointer&& itkImage) { image = std::move(itkImage); }
    
    CuboidShape getShape() const;
    void flip();
    void scale(int new_size_x, int new_size_y, int new_size_z);

    float getPixel(int x, int y, int z) const;
    void setPixel(int x, int y, int z, float value);
    void setRow(int row, int slice, const float* data);
    void setSlice(int sliceindex, const void* data);


    // Iterator wrapper for Image3D
    class Iterator {
    public: // Make members public so ConstIterator can access them
        itk::ImageRegionIterator<ImageType> itkIterator;
        ImageType::SizeType imageSize;
        ImageType::IndexType currentIndex;
        
    public:
        // Standard iterator typedefs
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = float;
        using pointer = float*;
        using reference = float&;
        
        // Constructor
        Iterator(ImageType::Pointer img, bool atEnd = false);
        
        // Copy constructor
        Iterator(const Iterator& other);
        
        // Assignment operator
        Iterator& operator=(const Iterator& other);
        
        // Dereference operators
        float& operator*();
        const float& operator*() const;
        float* operator->();
        const float* operator->() const;
        
        // Increment operators
        Iterator& operator++();
        Iterator operator++(int);
        
        // Comparison operators
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;
        
        // Quality of life methods
        void getCoordinates(int& x, int& y, int& z) const;
        ImageType::IndexType getIndex() const;
        size_t getLinearIndex() const;
        bool isAtEnd() const;
        float getValue() const;
        void setValue(float value);
        float getNeighbor(int dx, int dy, int dz) const;
        bool isValidCoordinate(int x, int y, int z) const;
        
        // New methods that return both coordinates and value
        PixelData getPixelData() const;
        PixelData getNeighborData(int dx, int dy, int dz) const;
    };
    
    // Const iterator wrapper
    class ConstIterator {
    private:
        itk::ImageRegionConstIterator<ImageType> itkIterator;
        ImageType::SizeType imageSize;
        ImageType::IndexType currentIndex;
        
    public:
        // Standard iterator typedefs
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const float;
        using pointer = const float*;
        using reference = const float&;
        
        // Constructor
        ConstIterator(ImageType::Pointer img, bool atEnd = false);
        
        // Copy constructor
        ConstIterator(const ConstIterator& other);
        
        // Constructor from non-const iterator
        ConstIterator(const Iterator& other);
        
        // Dereference operators
        const float& operator*() const;
        const float* operator->() const;
        
        // Increment operators
        ConstIterator& operator++();
        ConstIterator operator++(int);
        
        // Comparison operators
        bool operator==(const ConstIterator& other) const;
        bool operator!=(const ConstIterator& other) const;
        
        // Quality of life methods (const versions)
        void getCoordinates(int& x, int& y, int& z) const;
        ImageType::IndexType getIndex() const;
        size_t getLinearIndex() const;
        bool isAtEnd() const;
        float getValue() const;
        float getNeighbor(int dx, int dy, int dz) const;
        bool isValidCoordinate(int x, int y, int z) const;
        
        // New methods that return both coordinates and value
        PixelData getPixelData() const;
        PixelData getNeighborData(int dx, int dy, int dz) const;
    };
    
    // Container interface methods
    Iterator begin() {
        return Iterator(image);
    }
    
    Iterator end() {
        return Iterator(image, true);
    }
    
    ConstIterator begin() const {
        return ConstIterator(image);
    }
    
    ConstIterator end() const {
        return ConstIterator(image, true);
    }
    
    ConstIterator cbegin() const {
        return ConstIterator(image);
    }
    
    ConstIterator cend() const {
        return ConstIterator(image, true);
    }
};

struct PaddedImage{
    Image3D image;
    Padding padding;
};

struct ImageBuffer{
    Image3D image;
    BoxCoordWithPadding source;
    int interactedValue = 0; // TODO make this betterwidth of the imagebuffer which has been readFrom/writtenTo
};

struct ImageMaskPair{
    Image3D image;
    Image3D mask;
};

template<typename T>
struct CustomList{
    std::list<T> images; // since this class actually owns the data, perhaps keep track of if its still used or will be used in the future, each cube should be used only once
    T& find(int index){
            if (index >= images.size()) {
                throw std::out_of_range("Index out of range");
            }

            auto it = images.begin();
            std::advance(it, index); // move iterator to the desired index
            return *it;              // return reference to element
        }

    void deleteIndex(int index){
        if (index >= images.size()) {
            throw std::out_of_range("Index out of range");
        }

        auto it = images.begin();
        std::advance(it, index); // move iterator to the element
        images.erase(it);           // erase it from the list
    }
    size_t size(){
        return images.size();
    }
    void push_back(T image){
        images.push_back(std::move(image));
    }
};