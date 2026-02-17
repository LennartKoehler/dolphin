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
#include "dolphinbackend/ComplexData.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include <itkImage.h>
#include "dolphin/Image3D.h"

class PaddedImage;


namespace Postprocessor{

    void addCubeToImage(
        const Image3D& cube,
        Image3D& image
    );

    void insertCubeInImage(
        const Image3D& cube,
        const BoxCoord& cubeBox,
        Image3D& image,
        const BoxCoord& srcBox
    );

    
    using IteratorType = itk::ImageRegionIterator<ImageType>;
    struct ImageHelper {


        ImageType::Pointer image;
        ImageType::Pointer mask;
        IteratorType imageIt;
        IteratorType maskIt;
        
        ImageHelper(ImageType::Pointer img, ImageType::Pointer msk)
            : image(img), mask(msk),
                imageIt(img, img->GetLargestPossibleRegion()),
                maskIt(msk, msk->GetLargestPossibleRegion()) {}
    };


    void createWeightMasks(
        std::vector<ComplexData*>& masks,
        const ComplexData& frequencyFeatheringKernel,
        std::shared_ptr<IBackend> backend);


    Image3D addFeathering(
        std::vector<ImageMaskPair>& pair,
        int radius,
        double epsilon
    );
    void performWeightedBlending(
        std::vector<ImageHelper>& inputs,
        ImageType::Pointer output
    );
    
    void removePadding(Image3D& image, const Padding& padding);
    void cropToOriginalSize(Image3D& image, const CuboidShape& originalSize);

    void postprocessChannel(Image3D& image);
}