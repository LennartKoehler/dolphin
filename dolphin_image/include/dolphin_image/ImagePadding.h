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

#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/PaddingFillType.h"

namespace ImagePadding {

    Padding padToShape(Image3D& image3D, const CuboidShape& targetShape, PaddingFillType borderType);

    void padImage(Image3D& image, const Padding& padding, PaddingFillType borderType, float shapeScale = 1.0f);
    void padImageMirror(Image3D& image, const Padding& padding);
    void padImageZero(Image3D& image, const Padding& padding);
    void padImageLinear(Image3D& image, const Padding& padding);
    void padImageQuadratic(Image3D& image, const Padding& padding, float shapeScale = 1.0f);
    void padImageSinusoid(Image3D& image, const Padding& padding);
    void padImageGaussian(Image3D& image, const Padding& padding, float shapeScale = 1.0f);

    void expandToMinSize(Image3D& image, const CuboidShape& minSize);

} // namespace ImagePadding
