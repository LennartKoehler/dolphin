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

#include "dolphin/deconvolution/Preprocessor.h"
#include <stdexcept>
#include <cmath>
#include <itkConstantPadImageFilter.h>
#include <itkMirrorPadImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageDuplicator.h>


ComplexData Preprocessor::convertImageToComplexData(
    const Image3D& input) {

    CuboidShape shape = input.getShape();
    ComplexData result = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceComplexFull(shape);

    int index = 0;

    for (const auto& it : input) {
        result[index][0] = static_cast<real_t>(it);
        result[index][1] = 0.0;
        index ++;
    }

    return result;
}
RealData Preprocessor::convertImageToRealData(
    const Image3D& input) {

    CuboidShape shape = input.getShape();
    RealData result = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(shape);

    int index = 0;

    for (const auto& it : input) {
        result[index] = static_cast<real_t>(it);
        index ++;
    }

    return result;
}

Image3D Preprocessor::convertComplexDataToImage(
        const ComplexData& input){

    Image3D output(input.getSize(), 0.0f);

    int index = 0;
    for (auto& it : output) {
        real_t real = input[index][0];
        real_t imag = input[index][1];
        it = static_cast<float>(std::sqrt(real * real + imag * imag));
        index ++;
    }

    return output;
}

// TODO make this conversion part of the backend? because if data is on cuda device then this wont work
Image3D Preprocessor::convertRealDataToImage(
        const RealData& input){

    Image3D output(input.getSize(), 0.0f);

    int index = 0;
    for (auto& it : output) {
        real_t real = input[index];
        it = static_cast<float>(real);
        index ++;
    }

    return output;
}


/**
 * Zero-pad the image to the target size and fix the region to start at (0,0,0).
 * Returns the padded image with original data preserved and padding filled with 0.
 */
ImageType::Pointer zeroPadImage(const Image3D& image, const Padding& padding) {
    CuboidShape currentShape = image.getShape();

    ImageType::SizeType lowerExtendRegion;
    lowerExtendRegion[0] = padding.before.width;
    lowerExtendRegion[1] = padding.before.height;
    lowerExtendRegion[2] = padding.before.depth;

    ImageType::SizeType upperExtendRegion;
    upperExtendRegion[0] = padding.after.width;
    upperExtendRegion[1] = padding.after.height;
    upperExtendRegion[2] = padding.after.depth;

    using PadFilterType = itk::ConstantPadImageFilter<ImageType, ImageType>;
    typename PadFilterType::Pointer padFilter = PadFilterType::New();
    padFilter->SetPadLowerBound(lowerExtendRegion);
    padFilter->SetPadUpperBound(upperExtendRegion);
    padFilter->SetConstant(0.0f);
    padFilter->SetInput(image.getItkImage());
    padFilter->Update();

    ImageType::Pointer paddedImage = padFilter->GetOutput();

    // Force the region to start at (0,0,0)
    ImageType::SizeType paddedSize;
    paddedSize[0] = currentShape.width + padding.before.width + padding.after.width;
    paddedSize[1] = currentShape.height + padding.before.height + padding.after.height;
    paddedSize[2] = currentShape.depth + padding.before.depth + padding.after.depth;

    ImageType::IndexType zeroStart;
    zeroStart.Fill(0);

    ImageType::RegionType newRegion;
    newRegion.SetIndex(zeroStart);
    newRegion.SetSize(paddedSize);

    paddedImage->SetLargestPossibleRegion(newRegion);
    paddedImage->SetBufferedRegion(newRegion);
    paddedImage->SetRequestedRegion(newRegion);

    return paddedImage;
}

/**
 * Compute the normalized distance t for a single axis.
 * t = 0 at the original image boundary, t = 1 at the padded border.
 * Returns 0 if the pixel is inside the original image on this axis.
 */
float computeAxisDistance(long idx, long padBefore, long originalSize) {
    long padAfter_start = padBefore + originalSize;
    if (idx < padBefore && padBefore > 0) {
        return static_cast<float>(padBefore - idx) / static_cast<float>(padBefore);
    } else if (idx >= padAfter_start) {
        long padAfter = 0; // not available here; caller must handle
        return 0.0f; // fallback
    }
    return 0.0f;
}

/**
 * Apply a weighting function to the padded regions of a zero-padded image.
 * weightFn receives t (normalized distance from boundary, 0..1) and returns a weight [0..1].
 * The padded pixel value = boundaryPixelValue * weightFn(t).
 */
void applyPaddingWeight(
    ImageType::Pointer paddedImage,
    const Padding& padding,
    const CuboidShape& originalShape,
    std::function<float(float t)> weightFn)
{
    const int currentWidth = originalShape.width;
    const int currentHeight = originalShape.height;
    const int currentDepth = originalShape.depth;

    const int widthPaddingLeft = padding.before.width;
    const int heightPaddingTop = padding.before.height;
    const int depthPaddingBefore = padding.before.depth;

    const int widthPaddingRight = padding.after.width;
    const int heightPaddingBottom = padding.after.height;
    const int depthPaddingAfter = padding.after.depth;

    using IteratorType = itk::ImageRegionIteratorWithIndex<ImageType>;
    IteratorType it(paddedImage, paddedImage->GetLargestPossibleRegion());

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        ImageType::IndexType idx = it.GetIndex();

        // Check if this pixel is in the original image region
        bool inOriginalX = (idx[0] >= widthPaddingLeft) && (idx[0] < widthPaddingLeft + currentWidth);
        bool inOriginalY = (idx[1] >= heightPaddingTop) && (idx[1] < heightPaddingTop + currentHeight);
        bool inOriginalZ = (idx[2] >= depthPaddingBefore) && (idx[2] < depthPaddingBefore + currentDepth);

        if (inOriginalX && inOriginalY && inOriginalZ) {
            continue; // Inside the original image - keep the value as is
        }

        // Compute normalized distance t for each axis independently.
        // t = 0 at the original boundary, t = 1 at the padded border.
        float t_x = 0.0f;
        if (idx[0] < widthPaddingLeft && widthPaddingLeft > 0) {
            t_x = static_cast<float>(widthPaddingLeft - idx[0]) / static_cast<float>(widthPaddingLeft);
        } else if (idx[0] >= widthPaddingLeft + currentWidth && widthPaddingRight > 0) {
            t_x = static_cast<float>(idx[0] - (widthPaddingLeft + currentWidth - 1)) / static_cast<float>(widthPaddingRight);
        }

        float t_y = 0.0f;
        if (idx[1] < heightPaddingTop && heightPaddingTop > 0) {
            t_y = static_cast<float>(heightPaddingTop - idx[1]) / static_cast<float>(heightPaddingTop);
        } else if (idx[1] >= heightPaddingTop + currentHeight && heightPaddingBottom > 0) {
            t_y = static_cast<float>(idx[1] - (heightPaddingTop + currentHeight - 1)) / static_cast<float>(heightPaddingBottom);
        }

        float t_z = 0.0f;
        if (idx[2] < depthPaddingBefore && depthPaddingBefore > 0) {
            t_z = static_cast<float>(depthPaddingBefore - idx[2]) / static_cast<float>(depthPaddingBefore);
        } else if (idx[2] >= depthPaddingBefore + currentDepth && depthPaddingAfter > 0) {
            t_z = static_cast<float>(idx[2] - (depthPaddingBefore + currentDepth - 1)) / static_cast<float>(depthPaddingAfter);
        }

        // The overall fade factor: use the maximum t across all axes that are in padding.
        // This ensures a smooth fade even in corners where multiple axes are padded.
        float t = std::max({t_x, t_y, t_z});

        // Clamp t to [0, 1]
        t = std::clamp(t, 0.0f, 1.0f);

        // Get the value of the nearest boundary pixel in the original image
        long srcX = std::clamp<long>(idx[0], widthPaddingLeft, widthPaddingLeft + currentWidth - 1);
        long srcY = std::clamp<long>(idx[1], heightPaddingTop, heightPaddingTop + currentHeight - 1);
        long srcZ = std::clamp<long>(idx[2], depthPaddingBefore, depthPaddingBefore + currentDepth - 1);

        ImageType::IndexType srcIdx;
        srcIdx[0] = srcX;
        srcIdx[1] = srcY;
        srcIdx[2] = srcZ;

        float boundaryValue = paddedImage->GetPixel(srcIdx);
        float weight = weightFn(t);
        it.Set(boundaryValue * weight);
    }
}


void Preprocessor::padImage(Image3D& image, const Padding& padding, PaddingType paddingType, float shapeScale){
    if (paddingType == PaddingType::MIRROR) padImageMirror(image, padding);
    else if (paddingType == PaddingType::ZERO) padImageZero(image, padding);
    else if (paddingType == PaddingType::LINEAR) padImageLinear(image, padding);
    else if (paddingType == PaddingType::QUADRATIC) padImageQuadratic(image, padding, shapeScale);
    else if (paddingType == PaddingType::SINUSOID) padImageSinusoid(image, padding);
    else if (paddingType == PaddingType::GAUSSIAN) padImageGaussian(image, padding, shapeScale);
}


void Preprocessor::padImageLinear(Image3D& image, const Padding& padding){
    // Linear fade: weight = (1 - t)
    CuboidShape originalShape = image.getShape();
    ImageType::Pointer paddedImage = zeroPadImage(image, padding);
    applyPaddingWeight(paddedImage, padding, originalShape,
        [](float t) -> float {
            return 1.0f - t;
        });
    image.setItkImage(std::move(paddedImage));
}

void Preprocessor::padImageQuadratic(Image3D& image, const Padding& padding, float shapeScale){
    // Quadratic interpolation between a control point at value 0 (at distance = paddingSize * shapeScale)
    // and the nearest original image voxel (at distance = 0).
    // The parabola passes through (0, boundaryValue) and (paddingSize*shapeScale, 0),
    // with vertex at (paddingSize*shapeScale, 0).
    // weight = (1 - t)^2 when shapeScale = 1.0
    // With shapeScale > 1.0 the parabolas overlap and contributions are summed.
    CuboidShape originalShape = image.getShape();
    ImageType::Pointer paddedImage = zeroPadImage(image, padding);
    applyPaddingWeight(paddedImage, padding, originalShape,
        [shapeScale](float t) -> float {
            float t_scaled = t / shapeScale;
            float weight = 1.0f - t_scaled;
            return weight * weight; // (1 - t/scale)^2
        });
    image.setItkImage(std::move(paddedImage));
}

void Preprocessor::padImageSinusoid(Image3D& image, const Padding& padding){
    // Sinusoid: cos^2((pi/2) * t)
    // At t=0 (boundary): cos^2(0) = 1 (full value)
    // At t=1 (border): cos^2(pi/2) = 0
    CuboidShape originalShape = image.getShape();
    ImageType::Pointer paddedImage = zeroPadImage(image, padding);
    applyPaddingWeight(paddedImage, padding, originalShape,
        [](float t) -> float {
            float x = static_cast<float>(M_PI) * 0.5f * t;
            float c = std::cos(x);
            return c * c;
        });
    image.setItkImage(std::move(paddedImage));
}

void Preprocessor::padImageGaussian(Image3D& image, const Padding& padding, float shapeScale){
    // Gaussian: exp(-0.5 * (t * scale_factor)^2)
    // With shapeScale = 1.0, the Gaussian is 1/e at the padded border (t=1),
    // so scale_factor = sqrt(2) to make exp(-0.5 * (sqrt(2))^2) = exp(-1) = 1/e.
    CuboidShape originalShape = image.getShape();
    ImageType::Pointer paddedImage = zeroPadImage(image, padding);
    applyPaddingWeight(paddedImage, padding, originalShape,
        [shapeScale](float t) -> float {
            float t_scaled = t / shapeScale;
            return std::exp(-t_scaled * t_scaled); // exp(-t^2/scale^2): 1/e at t=shapeScale
        });
    image.setItkImage(std::move(paddedImage));
}

void Preprocessor::padImageMirror(Image3D& image, const Padding& padding){

    // Get current image dimensions using ITK
    CuboidShape currentShape = image.getShape();
    int currentDepth = currentShape.depth;
    int currentHeight = currentShape.height;
    int currentWidth = currentShape.width;

    // Set up padding amounts
    int depthPaddingBefore = padding.before.depth;
    int heightPaddingTop = padding.before.height;
    int widthPaddingLeft = padding.before.width;

    int depthPaddingAfter = padding.after.depth;
    int heightPaddingBottom = padding.after.height;
    int widthPaddingRight = padding.after.width;


    // Reflective padding using ITK MirrorPadImageFilter
    using MirrorPadFilterType = itk::MirrorPadImageFilter<ImageType, ImageType>;
    typename MirrorPadFilterType::Pointer mirrorPadFilter = MirrorPadFilterType::New();

    // Set up padding sizes
    ImageType::SizeType lowerExtendRegion;
    lowerExtendRegion[0] = widthPaddingLeft;
    lowerExtendRegion[1] = heightPaddingTop;
    lowerExtendRegion[2] = depthPaddingBefore;

    ImageType::SizeType upperExtendRegion;
    upperExtendRegion[0] = widthPaddingRight;
    upperExtendRegion[1] = heightPaddingBottom;
    upperExtendRegion[2] = depthPaddingAfter;

    mirrorPadFilter->SetPadLowerBound(lowerExtendRegion);
    mirrorPadFilter->SetPadUpperBound(upperExtendRegion);
    mirrorPadFilter->SetInput(image.getItkImage());
    mirrorPadFilter->Update();

    ImageType::Pointer paddedImage = mirrorPadFilter->GetOutput();

    // Force the region to start at (0,0,0) instead of negative coordinates
    ImageType::SizeType paddedSize;
    paddedSize[0] = currentWidth + widthPaddingLeft + widthPaddingRight;
    paddedSize[1] = currentHeight + heightPaddingTop + heightPaddingBottom;
    paddedSize[2] = currentDepth + depthPaddingBefore + depthPaddingAfter;

    ImageType::IndexType zeroStart;
    zeroStart.Fill(0);

    ImageType::RegionType newRegion;
    newRegion.SetIndex(zeroStart);
    newRegion.SetSize(paddedSize);

    // Reset all regions to start at (0,0,0)
    paddedImage->SetLargestPossibleRegion(newRegion);
    paddedImage->SetBufferedRegion(newRegion);
    paddedImage->SetRequestedRegion(newRegion);

    image.setItkImage(std::move(paddedImage));
}


void Preprocessor::padImageZero(Image3D& image, const Padding& padding){
    image.setItkImage(zeroPadImage(image,padding));
}

Padding Preprocessor::padToShape(Image3D& image, const CuboidShape& targetShape, PaddingType borderType){
    // Get current image dimensions using ITK
    CuboidShape currentShape = image.getShape();
    if (currentShape.width == 0 || currentShape.height == 0 || currentShape.depth == 0) {
        throw std::invalid_argument("Cannot pad empty image");
    }

    int currentDepth = currentShape.depth;
    int currentHeight = currentShape.height;
    int currentWidth = currentShape.width;

    // Calculate total padding needed
    int totalDepthPadding = targetShape.depth - currentDepth;
    int totalHeightPadding = targetShape.height - currentHeight;
    int totalWidthPadding = targetShape.width - currentWidth;

    int depthPaddingBefore = 0;
    int heightPaddingTop = 0;
    int widthPaddingLeft = 0;

    int depthPaddingAfter = 0;
    int heightPaddingBottom = 0;
    int widthPaddingRight = 0;

    // Handle depth padding (3D)
    if (totalDepthPadding > 0) {
        // Distribute padding: put extra padding at the end if odd
        depthPaddingBefore = totalDepthPadding / 2;
        depthPaddingAfter = totalDepthPadding - depthPaddingBefore;
    }

    // Handle 2D padding (width/height)
    if (totalHeightPadding > 0 || totalWidthPadding > 0) {
        heightPaddingTop = totalHeightPadding / 2;
        heightPaddingBottom = totalHeightPadding - heightPaddingTop;
        widthPaddingLeft = totalWidthPadding / 2;
        widthPaddingRight = totalWidthPadding - widthPaddingLeft;
    }

    Padding padding{
        CuboidShape{
            widthPaddingLeft,
            heightPaddingTop,
            depthPaddingBefore
        },
        CuboidShape{
            widthPaddingRight,
            heightPaddingBottom,
            depthPaddingAfter
        }
    };

    Preprocessor::padImage(image, padding, borderType);
    return padding;
}



void Preprocessor::expandToMinSize(Image3D& image, const CuboidShape& minSize) {
    // Get current image dimensions using ITK
    CuboidShape currentShape = image.getShape();
    if (currentShape.width == 0 || currentShape.height == 0 || currentShape.depth == 0) {
        return;
    }

    int currentDepth = currentShape.depth;
    int currentHeight = currentShape.height;
    int currentWidth = currentShape.width;

    // Calculate padding needed for each dimension
    int depthPadding = std::max(0, minSize.depth - currentDepth);
    int heightPadding = std::max(0, minSize.height - currentHeight);
    int widthPadding = std::max(0, minSize.width - currentWidth);

    // If no padding needed, return
    if (depthPadding == 0 && heightPadding == 0 && widthPadding == 0) {
        return;
    }

    // Use MirrorPadImageFilter for expansion (similar to the original BORDER_REFLECT_101)
    using MirrorPadFilterType = itk::MirrorPadImageFilter<ImageType, ImageType>;
    typename MirrorPadFilterType::Pointer mirrorPadFilter = MirrorPadFilterType::New();

    // Set up padding sizes - expand only at the end (like the original code)
    ImageType::SizeType lowerExtendRegion;
    lowerExtendRegion[0] = 0;              // left = 0 (no padding at left)
    lowerExtendRegion[1] = 0;              // top = 0 (no padding at top)
    lowerExtendRegion[2] = 0;              // front = 0 (no padding at front)

    ImageType::SizeType upperExtendRegion;
    upperExtendRegion[0] = widthPadding;   // right = all width padding
    upperExtendRegion[1] = heightPadding;  // bottom = all height padding
    upperExtendRegion[2] = depthPadding;   // back = all depth padding

    mirrorPadFilter->SetPadLowerBound(lowerExtendRegion);
    mirrorPadFilter->SetPadUpperBound(upperExtendRegion);
    mirrorPadFilter->SetInput(image.getItkImage());
    mirrorPadFilter->Update();

    ImageType::Pointer expandedImage = mirrorPadFilter->GetOutput();
    image.setItkImage(std::move(expandedImage));
}

