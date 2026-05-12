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
#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImageOperations.h"


namespace Postprocessor{

    // General image operations now delegate to ImageOperations namespace
    // These wrappers are kept for backward compatibility
    inline void addCubeToImage(const Image3D& cube, Image3D& image) { ImageOperations::addCubeToImage(cube, image); }
    inline void insertCubeInImage(const Image3D& cube, const BoxCoord& cubeBox, Image3D& image, const BoxCoord& srcBox) { ImageOperations::insertCubeInImage(cube, cubeBox, image, srcBox); }
    inline void removePadding(Image3D& image, const Padding& padding) { ImageOperations::removePadding(image, padding); }
    inline void cropToOriginalSize(Image3D& image, const CuboidShape& originalSize) { ImageOperations::cropToOriginalSize(image, originalSize); }
    inline void postprocessChannel(Image3D& image) { ImageOperations::normalizeChannel(image); }


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
        std::vector<RealData*>& masks,
        const ComplexData& frequencyFeatheringKernel,
        IBackend& backend);


    Image3D addFeathering(
        std::vector<ImageMaskPair>& pair,
        int radius,
        double epsilon
    );
    void performWeightedBlending(
        std::vector<ImageHelper>& inputs,
        ImageType::Pointer output
    );

}
