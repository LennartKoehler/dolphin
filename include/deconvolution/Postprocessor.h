#pragma once

#include <vector>
#include "backend/ComplexData.h"
#include "deconvolution/deconvolutionStrategies/ComputationalPlan.h"
#include "itkImage.h"
#include "Image3D.h"

class PaddedImage;


namespace Postprocessor{
    void insertCubeInImage(
        const Image3D& cube,
        const BoxCoord& cubeBox,
        Image3D& image,
        const BoxCoord& srcBox
    );
    

    Image3D addFeathering(
        std::vector<ImageMaskPair>& pair,
        int radius,
        double epsilon
    );
    
    void removePadding(Image3D& image, const Padding& padding);
    void cropToOriginalSize(Image3D& image, const RectangleShape& originalSize);

    void postprocessChannel(Image3D& image);
}