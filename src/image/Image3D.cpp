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

#include <iostream>
#include "dolphin/Image3D.h"

// Image3D Constructor implementations
Image3D::Image3D(const RectangleShape& shape) {
    image = ImageType::New();
    
    ImageType::SizeType size;
    size[0] = shape.width;
    size[1] = shape.height;
    size[2] = shape.depth;

    ImageType::IndexType start;
    start.Fill(0);

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(-1.0f);
}

Image3D::Image3D(const Image3D& other) {
    using DuplicatorType = itk::ImageDuplicator< ImageType >;
    if (other.image.IsNotNull()) {
        auto duplicator = DuplicatorType::New();
        duplicator->SetInputImage(other.image);
        duplicator->Update();
        image = duplicator->GetOutput();
    } else {
        image = ImageType::New();
    }
}

// Image3D Method implementations
Image3D Image3D::getInRange(float min, float max) const {
    // Create a new image with the same size as the input
    ImageType::Pointer resultImage = ImageType::New();
    
    resultImage->SetRegions(image->GetLargestPossibleRegion());
    resultImage->Allocate();
    resultImage->FillBuffer(0.0f);
    
    // Create iterators for input and output images
    itk::ImageRegionIterator<ImageType> inputIterator(image, image->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> outputIterator(resultImage, resultImage->GetLargestPossibleRegion());

    // Iterate through all pixels and create mask
    for (inputIterator.GoToBegin(), outputIterator.GoToBegin(); 
         !inputIterator.IsAtEnd(); 
         ++inputIterator, ++outputIterator) {
        float pixelValue = inputIterator.Get();
        // Set to 1.0 if pixel is in range, 0.0 otherwise
        if (pixelValue >= min && pixelValue <= max) {
            outputIterator.Set(1.0f);
        } else {
            outputIterator.Set(0.0f);
        }
    }
    
    return Image3D(std::move(resultImage));
}


Image3D Image3D::getSubimageCopy(const BoxCoord& coords) {
    // Use ITK's RegionOfInterestImageFilter to extract a subregion
    using ROIFilterType = itk::RegionOfInterestImageFilter<ImageType, ImageType>;
    using DuplicatorType = itk::ImageDuplicator<ImageType>;
    typename ROIFilterType::Pointer roiFilter = ROIFilterType::New(); 
    assert(!image.IsNull() && "Empty image");
    // Set up the region to extract
    ImageType::IndexType start;
    start[0] = coords.position.width;
    start[1] = coords.position.height;
    start[2] = coords.position.depth;
    
    ImageType::SizeType size;
    size[0] = coords.dimensions.width;
    size[1] = coords.dimensions.height;
    size[2] = coords.dimensions.depth;
    
    ImageType::RegionType desiredRegion;
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);
    
    roiFilter->SetRegionOfInterest(desiredRegion);
    roiFilter->SetInput(image);
    roiFilter->Update();

    auto duplicator = DuplicatorType::New();
    duplicator->SetInputImage(roiFilter->GetOutput());
    duplicator->Update();

    return Image3D(duplicator->GetOutput());
}

// Iterator implementation
Image3D::Iterator::Iterator(ImageType::Pointer img, bool atEnd) 
    : itkIterator(img, img->GetLargestPossibleRegion())
    , imageSize(img->GetLargestPossibleRegion().GetSize()) {
    if (atEnd) {
        itkIterator.GoToEnd();
    } else {
        itkIterator.GoToBegin();
        currentIndex = itkIterator.GetIndex();
    }
}

Image3D::Iterator::Iterator(const Iterator& other) 
    : itkIterator(other.itkIterator)
    , imageSize(other.imageSize)
    , currentIndex(other.currentIndex) {}

Image3D::Iterator& Image3D::Iterator::operator=(const Iterator& other) {
    if (this != &other) {
        itkIterator = other.itkIterator;
        imageSize = other.imageSize;
        currentIndex = other.currentIndex;
    }
    return *this;
}

float& Image3D::Iterator::operator*() {
    return itkIterator.Value();
}

const float& Image3D::Iterator::operator*() const {
    return const_cast<itk::ImageRegionIterator<ImageType>&>(itkIterator).Value();
}

float* Image3D::Iterator::operator->() {
    return &(itkIterator.Value());
}

const float* Image3D::Iterator::operator->() const {
    return &(const_cast<itk::ImageRegionIterator<ImageType>&>(itkIterator).Value());
}

Image3D::Iterator& Image3D::Iterator::operator++() {
    ++itkIterator;
    if (!itkIterator.IsAtEnd()) {
        currentIndex = itkIterator.GetIndex();
    }
    return *this;
}

Image3D::Iterator Image3D::Iterator::operator++(int) {
    Iterator tmp(*this);
    ++(*this);
    return tmp;
}

bool Image3D::Iterator::operator==(const Iterator& other) const {
    return (itkIterator.IsAtEnd() && other.itkIterator.IsAtEnd()) ||
           (!itkIterator.IsAtEnd() && !other.itkIterator.IsAtEnd() &&
            itkIterator.GetIndex() == other.itkIterator.GetIndex());
}

bool Image3D::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

void Image3D::Iterator::getCoordinates(int& x, int& y, int& z) const {
    if (!itkIterator.IsAtEnd()) {
        x = currentIndex[0];
        y = currentIndex[1];
        z = currentIndex[2];
    }
}

ImageType::IndexType Image3D::Iterator::getIndex() const {
    return currentIndex;
}

size_t Image3D::Iterator::getLinearIndex() const {
    if (itkIterator.IsAtEnd()) return 0;
    return currentIndex[2] * imageSize[0] * imageSize[1] + 
           currentIndex[1] * imageSize[0] + 
           currentIndex[0];
}

bool Image3D::Iterator::isAtEnd() const {
    return itkIterator.IsAtEnd();
}

float Image3D::Iterator::getValue() const {
    return itkIterator.Get();
}

void Image3D::Iterator::setValue(float value) {
    itkIterator.Set(value);
}

float Image3D::Iterator::getNeighbor(int dx, int dy, int dz) const {
    ImageType::IndexType neighborIndex;
    neighborIndex[0] = currentIndex[0] + dx;
    neighborIndex[1] = currentIndex[1] + dy;
    neighborIndex[2] = currentIndex[2] + dz;
    
    // Check bounds
    if (neighborIndex[0] >= 0 && neighborIndex[0] < static_cast<int>(imageSize[0]) &&
        neighborIndex[1] >= 0 && neighborIndex[1] < static_cast<int>(imageSize[1]) &&
        neighborIndex[2] >= 0 && neighborIndex[2] < static_cast<int>(imageSize[2])) {
        return itkIterator.GetImage()->GetPixel(neighborIndex);
    }
    return 0.0f; // or throw exception for out of bounds
}

bool Image3D::Iterator::isValidCoordinate(int x, int y, int z) const {
    return x >= 0 && x < static_cast<int>(imageSize[0]) &&
           y >= 0 && y < static_cast<int>(imageSize[1]) &&
           z >= 0 && z < static_cast<int>(imageSize[2]);
}

PixelData Image3D::Iterator::getPixelData() const {
    if (itkIterator.IsAtEnd()) {
        return PixelData(0.0f, -1, -1, -1);
    }
    return PixelData(itkIterator.Get(), currentIndex[0], currentIndex[1], currentIndex[2]);
}

PixelData Image3D::Iterator::getNeighborData(int dx, int dy, int dz) const {
    ImageType::IndexType neighborIndex;
    neighborIndex[0] = currentIndex[0] + dx;
    neighborIndex[1] = currentIndex[1] + dy;
    neighborIndex[2] = currentIndex[2] + dz;
    
    // Check bounds
    if (neighborIndex[0] >= 0 && neighborIndex[0] < static_cast<int>(imageSize[0]) &&
        neighborIndex[1] >= 0 && neighborIndex[1] < static_cast<int>(imageSize[1]) &&
        neighborIndex[2] >= 0 && neighborIndex[2] < static_cast<int>(imageSize[2])) {
        float value = itkIterator.GetImage()->GetPixel(neighborIndex);
        return PixelData(value, neighborIndex[0], neighborIndex[1], neighborIndex[2]);
    }
    return PixelData(0.0f, -1, -1, -1); // Invalid coordinates for out of bounds
}

// ConstIterator implementation
Image3D::ConstIterator::ConstIterator(ImageType::Pointer img, bool atEnd) 
    : itkIterator(img, img->GetLargestPossibleRegion())
    , imageSize(img->GetLargestPossibleRegion().GetSize()) {
    if (atEnd) {
        itkIterator.GoToEnd();
    } else {
        itkIterator.GoToBegin();
        currentIndex = itkIterator.GetIndex();
    }
}

Image3D::ConstIterator::ConstIterator(const ConstIterator& other) 
    : itkIterator(other.itkIterator)
    , imageSize(other.imageSize)
    , currentIndex(other.currentIndex) {}

Image3D::ConstIterator::ConstIterator(const Iterator& other) 
    : itkIterator(other.itkIterator.GetImage(), other.itkIterator.GetImage()->GetLargestPossibleRegion())
    , imageSize(other.imageSize)
    , currentIndex(other.currentIndex) {
    if (!other.itkIterator.IsAtEnd()) {
        itkIterator.SetIndex(currentIndex);
    } else {
        itkIterator.GoToEnd();
    }
}

const float& Image3D::ConstIterator::operator*() const {
    return itkIterator.Value();
}

const float* Image3D::ConstIterator::operator->() const {
    return &(itkIterator.Value());
}

Image3D::ConstIterator& Image3D::ConstIterator::operator++() {
    ++itkIterator;
    if (!itkIterator.IsAtEnd()) {
        currentIndex = itkIterator.GetIndex();
    }
    return *this;
}

Image3D::ConstIterator Image3D::ConstIterator::operator++(int) {
    ConstIterator tmp(*this);
    ++(*this);
    return tmp;
}

bool Image3D::ConstIterator::operator==(const ConstIterator& other) const {
    return (itkIterator.IsAtEnd() && other.itkIterator.IsAtEnd()) ||
           (!itkIterator.IsAtEnd() && !other.itkIterator.IsAtEnd() &&
            itkIterator.GetIndex() == other.itkIterator.GetIndex());
}

bool Image3D::ConstIterator::operator!=(const ConstIterator& other) const {
    return !(*this == other);
}

void Image3D::ConstIterator::getCoordinates(int& x, int& y, int& z) const {
    if (!itkIterator.IsAtEnd()) {
        x = currentIndex[0];
        y = currentIndex[1];
        z = currentIndex[2];
    }
}

ImageType::IndexType Image3D::ConstIterator::getIndex() const {
    return currentIndex;
}

size_t Image3D::ConstIterator::getLinearIndex() const {
    if (itkIterator.IsAtEnd()) return 0;
    return currentIndex[2] * imageSize[0] * imageSize[1] + 
           currentIndex[1] * imageSize[0] + 
           currentIndex[0];
}

bool Image3D::ConstIterator::isAtEnd() const {
    return itkIterator.IsAtEnd();
}

float Image3D::ConstIterator::getValue() const {
    return itkIterator.Get();
}

float Image3D::ConstIterator::getNeighbor(int dx, int dy, int dz) const {
    ImageType::IndexType neighborIndex;
    neighborIndex[0] = currentIndex[0] + dx;
    neighborIndex[1] = currentIndex[1] + dy;
    neighborIndex[2] = currentIndex[2] + dz;
    
    if (neighborIndex[0] >= 0 && neighborIndex[0] < static_cast<int>(imageSize[0]) &&
        neighborIndex[1] >= 0 && neighborIndex[1] < static_cast<int>(imageSize[1]) &&
        neighborIndex[2] >= 0 && neighborIndex[2] < static_cast<int>(imageSize[2])) {
        return itkIterator.GetImage()->GetPixel(neighborIndex);
    }
    return 0.0f;
}

bool Image3D::ConstIterator::isValidCoordinate(int x, int y, int z) const {
    return x >= 0 && x < static_cast<int>(imageSize[0]) &&
           y >= 0 && y < static_cast<int>(imageSize[1]) &&
           z >= 0 && z < static_cast<int>(imageSize[2]);
}

PixelData Image3D::ConstIterator::getPixelData() const {
    if (itkIterator.IsAtEnd()) {
        return PixelData(0.0f, -1, -1, -1);
    }
    return PixelData(itkIterator.Get(), currentIndex[0], currentIndex[1], currentIndex[2]);
}

PixelData Image3D::ConstIterator::getNeighborData(int dx, int dy, int dz) const {
    ImageType::IndexType neighborIndex;
    neighborIndex[0] = currentIndex[0] + dx;
    neighborIndex[1] = currentIndex[1] + dy;
    neighborIndex[2] = currentIndex[2] + dz;
    
    if (neighborIndex[0] >= 0 && neighborIndex[0] < static_cast<int>(imageSize[0]) &&
        neighborIndex[1] >= 0 && neighborIndex[1] < static_cast<int>(imageSize[1]) &&
        neighborIndex[2] >= 0 && neighborIndex[2] < static_cast<int>(imageSize[2])) {
        float value = itkIterator.GetImage()->GetPixel(neighborIndex);
        return PixelData(value, neighborIndex[0], neighborIndex[1], neighborIndex[2]);
    }
    return PixelData(0.0f, -1, -1, -1); // Invalid coordinates for out of bounds
}



// Legacy methods - these seem to be from an older version and need updating
void Image3D::flip() {
    // TODO: This method needs to be updated to work with ITK images
    // Current implementation seems to be using OpenCV Mat structure which doesn't match header
}

void Image3D::scale(int new_size_x, int new_size_y, int new_size_z) {

}

float Image3D::getPixel(int x, int y, int z) const {
    if (image.IsNull()) return 0.0f;
    
    ImageType::IndexType index = {x, y, z};
    return image->GetPixel(index);
}

void Image3D::setPixel(int x, int y, int z, float value) {
    if (image.IsNull()) return;
    
    ImageType::IndexType index = {x, y, z};
    image->SetPixel(index, value);
}


RectangleShape Image3D::getShape() const {
    if (image.IsNull()) return RectangleShape{0, 0, 0};
    
    ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    return RectangleShape{static_cast<int>(size[0]), static_cast<int>(size[1]), static_cast<int>(size[2])};
}

void Image3D::setSlice(int sliceindex, const void* data) {
    using PixelType = ImageType::PixelType;

    // Compute number of pixels in a slice
    ImageType::RegionType fullRegion = image->GetLargestPossibleRegion();
    ImageType::SizeType fullSize = fullRegion.GetSize();
    size_t slicePixels = fullSize[0] * fullSize[1];

    if (slicePixels == 0) {
        throw std::runtime_error("Data size does not match slice size");
    }

    // Define the slice region
    ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = sliceindex;

    ImageType::SizeType size;
    size[0] = fullSize[0];
    size[1] = fullSize[1];
    size[2] = 1;

    ImageType::RegionType sliceRegion;
    sliceRegion.SetIndex(start);
    sliceRegion.SetSize(size);

    // Iterator over the slice
    itk::ImageRegionIterator<ImageType> it(image, sliceRegion);

    PixelType* pixelData = (PixelType*)data;

    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++pixelData) {
        it.Set(*pixelData);
    }
}

void Image3D::setRow(int colindex, int sliceindex, const float* data) {
    using PixelType = ImageType::PixelType;

    // Compute number of pixels in a slice
    ImageType::RegionType fullRegion = image->GetLargestPossibleRegion();
    ImageType::SizeType fullSize = fullRegion.GetSize();
    size_t slicePixels = fullSize[0] * fullSize[1];

    if (slicePixels == 0) {
        throw std::runtime_error("Data size does not match slice size");
    }

    // Define the slice region
    ImageType::IndexType start;
    start[0] = 0;
    start[1] = colindex;
    start[2] = sliceindex;

    ImageType::SizeType size;
    size[0] = fullSize[0];
    size[1] = 1;
    size[2] = 1;

    ImageType::RegionType sliceRegion;
    sliceRegion.SetIndex(start);
    sliceRegion.SetSize(size);

    // Iterator over the slice
    itk::ImageRegionIterator<ImageType> it(image, sliceRegion);


    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++data) {
        it.Set(*data);
    }
}



