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

#include "deconvolution/Preprocessor.h"
#include <stdexcept>
#include "itkConstantPadImageFilter.h"
#include "itkMirrorPadImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageDuplicator.h"


ComplexData Preprocessor::convertImageToComplexData(
    const Image3D& input) {


    RectangleShape shape = input.getShape();
    ComplexData result = defaultBackendMemoryManager.allocateMemoryOnDevice(shape);

    int width = shape.width;
    int height = shape.height;
    int depth = shape.depth;
    
    int index = 0;
    for (const auto& it : input) {
        
        result.data[index][0] = static_cast<real_t>(it);
        result.data[index][1] = 0.0;
        index ++;
    }

    return result;
}

Image3D Preprocessor::convertComplexDataToImage(
        const ComplexData& input)
{
    const int width  = input.size.width;
    const int height = input.size.height;
    const int depth  = input.size.depth;

    Image3D output(RectangleShape(width, height, depth));

    const complex_t* in = input.data;
    int index = 0;
    for (auto& it : output) {
        real_t real = in[index][0];
        real_t imag = in[index][1];
        it = static_cast<float>(std::sqrt(real * real + imag * imag));
        index ++;

    }

    return output;
}


void Preprocessor::padImage(Image3D& image, const Padding& padding, PaddingType borderType){
    
    // Get current image dimensions using ITK
    RectangleShape currentShape = image.getShape();
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
    

    if (borderType == PaddingType::MIRROR) {
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
    // Choose appropriate ITK filter based on border type
    else {
        // Zero padding using ITK ConstantPadImageFilter
        using PadFilterType = itk::ConstantPadImageFilter<ImageType, ImageType>;
        typename PadFilterType::Pointer padFilter = PadFilterType::New();
        
        // Set up padding sizes (ITK uses [lower, upper] bounds for each dimension)
        ImageType::SizeType lowerExtendRegion;
        lowerExtendRegion[0] = widthPaddingLeft;
        lowerExtendRegion[1] = heightPaddingTop;
        lowerExtendRegion[2] = depthPaddingBefore;
        
        ImageType::SizeType upperExtendRegion;
        upperExtendRegion[0] = widthPaddingRight;
        upperExtendRegion[1] = heightPaddingBottom;
        upperExtendRegion[2] = depthPaddingAfter;
        
        padFilter->SetPadLowerBound(lowerExtendRegion);
        padFilter->SetPadUpperBound(upperExtendRegion);
        padFilter->SetConstant(0.0f);
        padFilter->SetInput(image.getItkImage());
        padFilter->Update();
        
        ImageType::Pointer paddedImage = padFilter->GetOutput();
        
        // Force the region to start at (0,0,0) instead of negative coordinates
        ImageType::SizeType paddedSize = paddedImage->GetLargestPossibleRegion().GetSize();
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
}


Padding Preprocessor::padToShape(Image3D& image, const RectangleShape& targetShape, PaddingType borderType){
    // Get current image dimensions using ITK
    RectangleShape currentShape = image.getShape();
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
        RectangleShape{
            widthPaddingLeft,
            heightPaddingTop,
            depthPaddingBefore
        },
        RectangleShape{
            widthPaddingRight,
            heightPaddingBottom,
            depthPaddingAfter
        }
    };

    Preprocessor::padImage(image, padding, borderType);
    return padding;
}



void Preprocessor::expandToMinSize(Image3D& image, const RectangleShape& minSize) {
    // Get current image dimensions using ITK
    RectangleShape currentShape = image.getShape();
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

