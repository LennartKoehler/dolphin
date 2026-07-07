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

#include "dolphin_image/Image3D.h"
#include <functional>
#include <itkTestingComparisonImageFilter.h>
#include <cmath>
#include <cstring>

Image3D::Image3D(const CuboidShape& shape, float fillValue) {
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
    image->FillBuffer(fillValue);
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


Image3D::Image3D(Image3D&& other){
    image = other.image;

}

Image3D& Image3D::operator=(Image3D&& other) noexcept {

    if (this != &other){
        image = other.image;
    }
    return *this;
}

Image3D& Image3D::operator=(const Image3D& other) {
    using DuplicatorType = itk::ImageDuplicator< ImageType >;
    if (this != &other){
        if (other.image.IsNotNull()) {
            auto duplicator = DuplicatorType::New();
            duplicator->SetInputImage(other.image);
            duplicator->Update();
            image = duplicator->GetOutput();
        } else {
            image = ImageType::New();
        }
    }
    return *this;
}

bool Image3D::isEqual(const Image3D& other, float tolerance) const{

    // Handle null images
    if (image.IsNull() && other.image.IsNull()) {
        return true;
    }
    if (image.IsNull() || other.image.IsNull()) {
        return false;
    }

    // Images with different sizes are not equal
    if (getShape() != other.getShape()) {
        return false;
    }

    // Pre-pass: check for mismatched NaN values.
    // itk::Testing::ComparisonImageFilter skips NaN pixels because
    // (NaN > threshold) is always false in IEEE 754, so it would
    // incorrectly consider a NaN pixel and a valid pixel as "equal".
    {
        itk::ImageRegionConstIterator<ImageType> itThis(image, image->GetLargestPossibleRegion());
        itk::ImageRegionConstIterator<ImageType> itOther(other.image, other.image->GetLargestPossibleRegion());
        for (itThis.GoToBegin(), itOther.GoToBegin(); !itThis.IsAtEnd(); ++itThis, ++itOther) {
            const bool thisIsNaN = std::isnan(itThis.Get());
            const bool otherIsNaN = std::isnan(itOther.Get());
            if (thisIsNaN != otherIsNaN) {
                return false;
            }
        }
    }

    using CompareFilterType = itk::Testing::ComparisonImageFilter<ImageType, ImageType>;

    auto compare = CompareFilterType::New();
    compare->SetValidInput(this->getItkImage());
    compare->SetTestInput(other.getItkImage());

    compare->SetDifferenceThreshold(tolerance);  // tolerance per pixel

    compare->Update();

    const auto numberOfDifferentPixels = compare->GetNumberOfPixelsWithDifferences();
    return numberOfDifferentPixels == 0;
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
    start[0] = static_cast<itk::IndexValueType>(coords.position.width);
    start[1] = static_cast<itk::IndexValueType>(coords.position.height);
    start[2] = static_cast<itk::IndexValueType>(coords.position.depth);

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

void Image3D::Iterator::getCoordinates(size_t& x, size_t& y, size_t& z) const {
    if (!itkIterator.IsAtEnd()) {
        x = static_cast<size_t>(currentIndex[0]);
        y = static_cast<size_t>(currentIndex[1]);
        z = static_cast<size_t>(currentIndex[2]);
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

bool Image3D::Iterator::isValidCoordinate(size_t x, size_t y, size_t z) const {
    return x < imageSize[0] && y < imageSize[1] && z < imageSize[2];
}

PixelData Image3D::Iterator::getPixelData() const {
    if (itkIterator.IsAtEnd()) {
        return PixelData(0.0f, SIZE_MAX, SIZE_MAX, SIZE_MAX);
    }
    return PixelData(itkIterator.Get(), static_cast<size_t>(currentIndex[0]), static_cast<size_t>(currentIndex[1]), static_cast<size_t>(currentIndex[2]));
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
    return PixelData(0.0f, SIZE_MAX, SIZE_MAX, SIZE_MAX); // Invalid coordinates for out of bounds
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

void Image3D::ConstIterator::getCoordinates(size_t& x, size_t& y, size_t& z) const {
    if (!itkIterator.IsAtEnd()) {
        x = static_cast<size_t>(currentIndex[0]);
        y = static_cast<size_t>(currentIndex[1]);
        z = static_cast<size_t>(currentIndex[2]);
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

bool Image3D::ConstIterator::isValidCoordinate(size_t x, size_t y, size_t z) const {
    return x < imageSize[0] && y < imageSize[1] && z < imageSize[2];
}

PixelData Image3D::ConstIterator::getPixelData() const {
    if (itkIterator.IsAtEnd()) {
        return PixelData(0.0f, SIZE_MAX, SIZE_MAX, SIZE_MAX);
    }
    return PixelData(itkIterator.Get(), static_cast<size_t>(currentIndex[0]), static_cast<size_t>(currentIndex[1]), static_cast<size_t>(currentIndex[2]));
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
    return PixelData(0.0f, SIZE_MAX, SIZE_MAX, SIZE_MAX); // Invalid coordinates for out of bounds
}



// Legacy methods - these seem to be from an older version and need updating
void Image3D::flip() {
    // TODO: This method needs to be updated to work with ITK images
}

float Image3D::getMax() const {

    if (image.IsNull()) return 0.0;

    const auto region = image->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const auto startIndex = region.GetIndex();
    float max = 0.0;

    itk::ImageRegionConstIterator<ImageType> it(image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float val = it.Get();
        if (val > max){
            max = val;
        }
    }
    return max;
}

CuboidShape Image3D::getRegionLargerThreshold(float threshold) const{
    if (image.IsNull()) return CuboidShape{0, 0, 0};

    const auto region = image->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const auto startIndex = region.GetIndex();

    int64_t minX = static_cast<int64_t>(size[0]);
    int64_t minY = static_cast<int64_t>(size[1]);
    int64_t minZ = static_cast<int64_t>(size[2]);
    int64_t maxX = -1;
    int64_t maxY = -1;
    int64_t maxZ = -1;

    itk::ImageRegionConstIterator<ImageType> it(image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (it.Get() > threshold) {
            auto idx = it.GetIndex();
            int64_t x = static_cast<int64_t>(idx[0] - startIndex[0]);
            int64_t y = static_cast<int64_t>(idx[1] - startIndex[1]);
            int64_t z = static_cast<int64_t>(idx[2] - startIndex[2]);
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            minZ = std::min(minZ, z);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
            maxZ = std::max(maxZ, z);
        }
    }

    if (maxX < 0) return CuboidShape{0, 0, 0};

    return CuboidShape{
        static_cast<size_t>(maxX - minX + 1),
        static_cast<size_t>(maxY - minY + 1),
        static_cast<size_t>(maxZ - minZ + 1)
    };
}

void Image3D::scale(size_t new_size_x, size_t new_size_y, size_t new_size_z) {

}

float Image3D::getPixel(size_t x, size_t y, size_t z) const {
    if (image.IsNull()) return 0.0f;

    ImageType::IndexType index = {static_cast<itk::IndexValueType>(x), static_cast<itk::IndexValueType>(y), static_cast<itk::IndexValueType>(z)};
    return image->GetPixel(index);
}

void Image3D::setPixel(size_t x, size_t y, size_t z, float value) {
    if (image.IsNull()) return;

    ImageType::IndexType index = {static_cast<itk::IndexValueType>(x), static_cast<itk::IndexValueType>(y), static_cast<itk::IndexValueType>(z)};
    image->SetPixel(index, value);
}

float& Image3D::operator[](size_t offset){return image->GetBufferPointer()[offset];}

CuboidShape Image3D::getShape() const {
    if (image.IsNull()) return CuboidShape{0, 0, 0};

    ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    return CuboidShape{size[0], size[1], size[2]};
}


void Image3D::executeOperations(std::vector<std::reference_wrapper<IImageOperation>>& operations){
    assert(!image.IsNull());

    const auto region = image->GetLargestPossibleRegion();
    itk::ImageRegionIterator<ImageType> it(image, region);
    size_t iterator = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        for (IImageOperation& operation : operations){
            operation(iterator, it.Value());
            iterator++;
        }
    }
}
void Image3D::executeOperations(std::vector<std::reference_wrapper<IConstImageOperation>>& operations)const{
    assert(!image.IsNull());

    const auto region = image->GetLargestPossibleRegion();
    itk::ImageRegionIterator<ImageType> it(image, region);
    size_t iterator = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        for (IConstImageOperation& operation : operations){
            operation(iterator, it.Value());
            iterator++;
        }
    }
}

void Image3D::setSlice(size_t sliceindex, const void* data) {
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
    start[2] = static_cast<itk::IndexValueType>(sliceindex);

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

void Image3D::setRow(size_t colindex, size_t sliceindex, const float* data) {
    ImageType::RegionType fullRegion = image->GetLargestPossibleRegion();
    ImageType::SizeType fullSize = fullRegion.GetSize();
    size_t width = fullSize[0];
    size_t slicePixels = width * fullSize[1];

    if (slicePixels == 0) {
        throw std::runtime_error("Data size does not match slice size");
    }

    size_t offset = colindex * width + sliceindex * slicePixels;
    std::memcpy(image->GetBufferPointer() + offset, data, width * sizeof(float));
}


void LazyImage3D::update(){
    assert(!image.IsNull());

    const auto region = image->GetLargestPossibleRegion();
    itk::ImageRegionIterator<ImageType> it(image, region);
    size_t iterator = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        for (IImageOperation& operation : deferredOperations){
            operation(iterator, it.Value());
            iterator++;
        }
    }
}
