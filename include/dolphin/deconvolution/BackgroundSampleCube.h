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

#include <optional>
#include <vector>

#include "dolphin_image/BackgroundStatistics.h"
#include "dolphin_image/Types/BoxCoord.h"

class ReaderHandler;

// Background level for a whole run, measured on ONE cube of the image.
//
// Why one cube: the background is a property of the whole image, but reading
// the whole image only to estimate a single scalar doubles the IO time. One
// cube is representative and cheap. The cube is chosen so the estimate is not
// skewed:
//   - a cube whose padded box lies entirely inside the image is preferred, so
//     the padding fill (which fades towards zero) is interior content;
//   - among those, the one closest to the start in z, then y, then x is used
//     (deterministic, and near the beginning of the file for slow readers);
//   - when no cube is fully interior (a single cube larger than the image), the
//     first cube is used and its padded border is excluded from the estimate.
// The estimate is taken on the cube as read. Cube padding only re-reads image
// content (or a fill that is excluded), so it does not change the value.
float estimateBackgroundFromSampleCube(
    const ReaderHandler& reader,
    const BoxCoordWithPadding& cubes);
