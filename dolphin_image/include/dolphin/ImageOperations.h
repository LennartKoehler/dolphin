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

#include "dolphin/Image3D.h"

namespace ImageOperations {

    void insertCubeInImage(
        const Image3D& cube,
        const BoxCoord& cubeBox,
        Image3D& image,
        const BoxCoord& srcBox
    );

    void addCubeToImage(
        const Image3D& cube,
        Image3D& image
    );

    void removePadding(Image3D& image, const Padding& padding);
    void cropToOriginalSize(Image3D& image, const CuboidShape& originalSize);

    void normalizeChannel(Image3D& image);

} // namespace ImageOperations
