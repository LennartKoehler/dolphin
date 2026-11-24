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

#include <cstddef>

#include "dolphin_image/BackgroundStatistics.h"
#include "dolphin_image/Types/BoxCoord.h"

class ImageReader;
class ReaderHandler;

// Run-wide background level (camera offset + constant stray light) for one
// channel, resolved once so every cube of a run is deconvolved against the
// same level (estimating per cube would differ between neighbours and leave
// visible tile seams). Two strategies, selected by background_strategy:
//
// - whole_image: streams the ENTIRE image through the reader in z-chunks (the
//   image is never held in RAM) and accumulates one histogram over all voxels.
//   Accurate, but reads the image twice, once per statistics pass.
//
// - global: measures ONE cube of the image. Cheap (a single cube read); the
//   cube is assumed representative of the image's background. The caller
//   picks the cube (currently the first of the run).
//
// Both estimate the mode of the dark cluster below a small percentile bound
// (see BackgroundStatistics) over UNPADDED reads: background reads are
// unpadded, so synthetic padding fill cannot bias them.
//
// The reader must be configured and its memory cap applied by the caller.
// chunkHint is the (unpadded) cube shape of the run; the z-chunk height is
// scaled down so a chunk stays within it.
float estimateImageBackground(ImageReader& reader,
                              const CuboidShape& imageShape,
                              const CuboidShape& chunkHint,
                              size_t readerThreads);

// Background level measured on one cube of the image, as read by the handler.
// Throws std::runtime_error when the cube contains no countable voxels.
float estimateBackgroundFromSampleCube(const ReaderHandler& reader,
                                       const BoxCoordWithPadding& cube);
