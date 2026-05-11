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
#include <itkImageDuplicator.h>
#include <itkSubtractImageFilter.h>

// insertCubeInImage, addCubeToImage, removePadding, cropToOriginalSize, and postprocessChannel
// have been moved to src/image/ImageOperations.cpp (ImageOperations namespace).
// The Postprocessor namespace now delegates to ImageOperations via inline functions in the header.

//binary masks are converted into masks whose weight resembles (1 - distance to label) because label is 1 and backgorund 0
// the feathering kernel is used for convolution with the binary mask. This creates a blue at the edge
// then all masks are summed to one so that the total weight is one
void Postprocessor::createWeightMasks(
    std::vector<RealData*>& masks,
    const ComplexData& frequencyFeatheringKernel,
    IBackend& backend
){
    for (RealData* mask_p : masks){
        RealData& mask = *mask_p;
        // convolution
        ComplexData maskComplex = backend.getMemoryManager().allocateMemoryOnDeviceComplex(mask.getSize());
        backend.getComputeManager().forwardFFT(mask, maskComplex);
        backend.getComputeManager().complexMultiplication(maskComplex,  frequencyFeatheringKernel, maskComplex);
        backend.getComputeManager().backwardFFT(maskComplex, mask);
    }

    real_t** masksarray = backend.getMemoryManager().createDataArray(masks);
    backend.getComputeManager().sumToOne(masksarray, masks.size(), masks[0]->getSize().getVolume());
}



// this also merges the images
// this could be implemented as another option for createWeight masks, and not do the merging
// ERROR the last mask cannot be seen as background, as this will not be normalized with more than two masks.
// feathering: last mask can not be seen as the background, because if there is a section where 3 masks meet, the 2 earlier masks would have weight ~0.5 and therefore the last mask would get 1 - 0.5 - 0.5 = 0. It should however be 1/3 for all. Only if there are only 2 masks can i do this.
// unused anyway
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

