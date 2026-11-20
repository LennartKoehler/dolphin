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

#include "dolphin_image/BackgroundStatistics.h"

class ImageReader;

// Whole-image background (mean of the dimmest decile) for one channel.
//
// Reads the image in z-chunks through the reader (so it works for images far
// larger than RAM), applies the same padding as the deconvolution cubes, and
// deliberately drops the padded border when accumulating (see
// forEachCountedVoxel): at the plan stage there is no information yet about
// which borders are inside the image and which are outside, and the padding
// fill fades towards zero, which would bias the estimate.
//
// chunkHint is the (unpadded) cube shape of the run; the chunk height is scaled
// down so a chunk stays within it. The reader must be configured and its memory
// cap applied by the caller.
float estimateImageBackground(ImageReader& reader,
                              const CuboidShape& imageShape,
                              const CuboidShape& chunkHint,
                              PaddingFillType fill,
                              size_t readerThreads);