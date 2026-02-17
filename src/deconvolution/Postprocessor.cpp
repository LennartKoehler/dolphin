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

#include "dolphin/deconvolution/Postprocessor.h"
#include <stdexcept>
#include <functional>
#include "dolphin/HelperClasses.h"
#include <itkImage.h>
#include <itkDanielssonDistanceMapImageFilter.h>
// #include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkPasteImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkCropImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>


void Postprocessor::insertCubeInImage(
    const Image3D& cube,
    const BoxCoord& cubeBox,
    Image3D& image,
    const BoxCoord& srcBox
) {
    using PasteFilterType = itk::PasteImageFilter<ImageType>;
    using ExtractFilterType = itk::ExtractImageFilter<ImageType, ImageType>;
    
    // Extract the region from the cube based on cubeBox
    ImageType::RegionType extractRegion;
    ImageType::IndexType extractStart;
    extractStart[0] = cubeBox.position.width;
    extractStart[1] = cubeBox.position.height;
    extractStart[2] = cubeBox.position.depth;
    
    ImageType::SizeType extractSize;
    extractSize[0] = cubeBox.dimensions.width;
    extractSize[1] = cubeBox.dimensions.height;
    extractSize[2] = cubeBox.dimensions.depth;
    
    extractRegion.SetIndex(extractStart);
    extractRegion.SetSize(extractSize);
    
    auto extractFilter = ExtractFilterType::New();
    extractFilter->SetInput(cube.getItkImage());
    extractFilter->SetExtractionRegion(extractRegion);
    extractFilter->SetDirectionCollapseToStrategy(ExtractFilterType::DIRECTIONCOLLAPSETOGUESS);
    extractFilter->Update();
    
    // Set up the destination region in the target image
    ImageType::IndexType destIndex;
    destIndex[0] = srcBox.position.width;
    destIndex[1] = srcBox.position.height;
    destIndex[2] = srcBox.position.depth;
    
    // Use PasteImageFilter to paste the extracted region into the target image
    auto pasteFilter = PasteFilterType::New();
    pasteFilter->SetSourceImage(extractFilter->GetOutput());
    pasteFilter->SetDestinationImage(image.getItkImage());
    pasteFilter->SetDestinationIndex(destIndex);
    pasteFilter->SetSourceRegion(extractFilter->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->Update();
    
    // Update the image with the result
    image.setItkImage(pasteFilter->GetOutput());
}
void Postprocessor::addCubeToImage(
    const Image3D& cube,
    Image3D& image
) {

    // Use PasteImageFilter to paste the extracted region into the target image
    using AddFilterType = itk::AddImageFilter<ImageType, ImageType, ImageType>;

    auto addFilter = AddFilterType::New();
    addFilter->SetInput1(cube.getItkImage());
    addFilter->SetInput2(image.getItkImage());
    addFilter->Update();

    ImageType::Pointer temp = addFilter->GetOutput();
    image.setItkImage(temp); 
}

//binary masks are converted into masks whose weight resembles (1 - distance to label) because label is 1 and backgorund 0
// the feathering kernel is used for convolution with the binary mask. This creates a blue at the edge
// then all masks are summed to one so that the total weight is one
void Postprocessor::createWeightMasks(
    std::vector<ComplexData*>& masks,
    const ComplexData& frequencyFeatheringKernel,
    std::shared_ptr<IBackend> backend
){
    for (ComplexData* mask_p : masks){
        ComplexData& mask = *mask_p;
        // convolution
        backend->getDeconvManager().forwardFFT(mask, mask);
        backend->getDeconvManager().complexMultiplication(mask,  frequencyFeatheringKernel, mask);
        backend->getDeconvManager().backwardFFT(mask, mask);
    }

    complex_t** masksarray = backend->getMemoryManager().createDataArray(masks);
    backend->getDeconvManager().sumToOneReal(masksarray, masks.size(), masks[0]->size.getVolume());
}



// this also merges the images
// this could be implemented as another option for createWeight masks, and not do the merging
Image3D Postprocessor::addFeathering(
    std::vector<ImageMaskPair>& pairs,
    int radius,
    double epsilon
) {
    using PixelType = float;
    constexpr unsigned int Dimension = 3;
    using ImageType = itk::Image<PixelType, Dimension>;
    using DistanceFilterType = itk::DanielssonDistanceMapImageFilter<ImageType, ImageType>;

    std::vector<ImageHelper> itkImageMasks;
    for (auto& pair : pairs) {

        itkImageMasks.emplace_back(pair.image.getItkImage(), pair.mask.getItkImage());
    }

    using DuplicatorType = itk::ImageDuplicator<ImageType>;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(itkImageMasks[0].image);
    duplicator->Update();
    ImageType::Pointer output = duplicator->GetOutput();

    // Process distance maps for all masks
    for (size_t i = 0; i < itkImageMasks.size() - 1; ++i) {// the last one is the "background"
    // for (auto& imagemask : itkImageMasks) {
        ImageHelper& imagemask = itkImageMasks[i];
        // Create the distance filter
        DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
        distanceFilter->SetInput(imagemask.mask);
        distanceFilter->Update();

        ImageType::Pointer distanceImage = distanceFilter->GetOutput();

        // Clip distances at 'radius' and convert back to 0-1 gradient
        itk::ImageRegionIterator<ImageType> it(distanceImage, distanceImage->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            float dist = it.Get();
            if (dist > radius)
                dist = radius;
            it.Set(1.0f - dist / radius); // gradient: 1 at mask, 0 at radius
        }
        
        imagemask.mask = distanceImage;
        // Recreate the mask iterator since we replaced the mask image
        imagemask.maskIt = IteratorType(imagemask.mask, imagemask.mask->GetLargestPossibleRegion());
    }



    // last mask can be seen as the background and is just 1 - all other masks
    // this is wrong, 
    using SubtractFilterType = itk::SubtractImageFilter<ImageType, ImageType, ImageType>;

    ImageHelper& background = itkImageMasks.back();
    for (background.maskIt.GoToBegin(); !background.maskIt.IsAtEnd(); ++background.maskIt)
    {
        background.maskIt.Set(1.0);  // set every pixel to 1
    }

    for (size_t i = 0; i < itkImageMasks.size() - 1; i++){
        ImageHelper& otherImageMask = itkImageMasks[i];
        SubtractFilterType::Pointer subtractFilter = SubtractFilterType::New();

        subtractFilter->SetInput1(background.mask);
        subtractFilter->SetInput2(otherImageMask.mask);
        subtractFilter->Update();
        background.mask = subtractFilter->GetOutput();
        
    }
    background.maskIt = IteratorType(background.mask, background.mask->GetLargestPossibleRegion());
    
    performWeightedBlending(
        itkImageMasks,
        output
    );

    // background.mask->DisconnectPipeline();
    return Image3D(std::move(output));

}
void Postprocessor::performWeightedBlending(
    std::vector<ImageHelper>& itkImageMasks,
    ImageType::Pointer output
){

    // Perform weighted blending
    IteratorType outIt(output, output->GetLargestPossibleRegion());
    
    for (auto& image : itkImageMasks) {
        image.imageIt.GoToBegin();
        image.maskIt.GoToBegin();
    }
    outIt.GoToBegin();

    std::vector<float> weights;
    std::vector<float> values;
    weights.reserve(itkImageMasks.size());
    values.reserve(itkImageMasks.size());

    while (!outIt.IsAtEnd()) {
        weights.clear();
        values.clear();
        float sum = 1e-6f; // Initialize sum to avoid division by zero

        for (auto& image : itkImageMasks) {
            float weight = image.maskIt.Get();
            values.push_back(image.imageIt.Get());
            weights.push_back(weight);
            sum += weight;

            ++image.imageIt;
            ++image.maskIt;
        }
        
        float resultValue = 0;
        if (sum > 0) {
            for (size_t i = 0; i < values.size(); i++) {
                resultValue += (weights[i] / sum) * values[i];
            }
        }
        outIt.Set(resultValue);
        ++outIt;
    }
}

void Postprocessor::removePadding(Image3D& image, const Padding& padding) {
    using CropFilterType = itk::CropImageFilter<ImageType, ImageType>;
    
    auto cropFilter = CropFilterType::New();
    cropFilter->SetInput(image.getItkImage());
    
    // Set the amount to crop from each side
    ImageType::SizeType lowerBound;
    lowerBound[0] = padding.before.width;  // x direction
    lowerBound[1] = padding.before.height; // y direction
    lowerBound[2] = padding.before.depth;  // z direction
    
    ImageType::SizeType upperBound;
    upperBound[0] = padding.after.width;   // x direction
    upperBound[1] = padding.after.height;  // y direction
    upperBound[2] = padding.after.depth;   // z direction
    
    cropFilter->SetLowerBoundaryCropSize(lowerBound);
    cropFilter->SetUpperBoundaryCropSize(upperBound);
    cropFilter->Update();
    
    // Update the image with the cropped result
    image.setItkImage(std::move(cropFilter->GetOutput()));
}

void Postprocessor::cropToOriginalSize(Image3D& image, const CuboidShape& originalSize) {
    CuboidShape currentSize = image.getShape();
    
    // Calculate how much to crop from each dimension
    CuboidShape cropAmount(std::max(0, currentSize.width - originalSize.width),
                             std::max(0, currentSize.height - originalSize.height),
                             std::max(0, currentSize.depth - originalSize.depth));
    
    // For symmetric cropping, distribute evenly between start and end
    CuboidShape cropStart(cropAmount.width / 2, cropAmount.height / 2, cropAmount.depth / 2);
    CuboidShape cropEnd = cropAmount - cropStart;
    
    Padding padding{cropStart, cropEnd};
    removePadding(image, padding);
}

void Postprocessor::postprocessChannel(Image3D& image){
    // Global normalization of the merged volume using ITK iterators
    double global_max_val = 0.0;
    double global_min_val = std::numeric_limits<double>::max();
    
    // Find global min and max using ITK iterator
    for (auto it = image.begin(); it != image.end(); ++it) {
        float pixelValue = *it;
        if (pixelValue < 0) {
            *it = 0.0f;  // Threshold negative values to zero
            pixelValue = 0.0f;
        }
        global_max_val = std::max(global_max_val, static_cast<double>(pixelValue));
        global_min_val = std::min(global_min_val, static_cast<double>(pixelValue));
    }
    
    // Normalize to [0,1] range and threshold small values
    float epsilon = 1e-6f;
    double scale = 1.0 / (global_max_val - global_min_val);
    
    for (auto it = image.begin(); it != image.end(); ++it) {
        float normalizedValue = static_cast<float>((*it - global_min_val) * scale);
        *it = (normalizedValue < epsilon) ? 0.0f : normalizedValue;
    }
}