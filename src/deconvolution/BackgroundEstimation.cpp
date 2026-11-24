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

#include "dolphin/deconvolution/BackgroundEstimation.h"

#include <algorithm>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "dolphin_image/IO/ReaderWriter.h"

namespace {

// Z-chunk height: tall enough to split the reader work across `threads`, but
// each chunk stays within the memory the run has for one cube.
size_t chunkDepth(size_t imageDepth, const CuboidShape& chunkHint, size_t threads) {
    const size_t wanted = std::max<size_t>(1, threads) * 2;
    const size_t perChunk = std::max<size_t>(1, (imageDepth + wanted - 1) / wanted);
    const size_t limit = std::max<size_t>(1, chunkHint.depth);
    return std::min(perChunk, limit);
}

// Runs one streaming pass over all z-chunks of the image: fn(image) is called
// per chunk with the unpadded chunk (only real image voxels).
template <typename Fn>
void forEachImageChunk(ImageReader& reader, const CuboidShape& imageShape,
                       const CuboidShape& chunkHint, size_t threads, Fn&& fn) {
    const size_t depth = chunkDepth(imageShape.depth, chunkHint, threads);

    for (size_t z = 0; z < imageShape.depth; z += depth) {
        const size_t zEnd = std::min(z + depth, imageShape.depth);
        // Unpadded read: every voxel of the chunk is real image content, so
        // no synthetic fill can bias the estimate.
        const BoxCoord box{CuboidPosition{0, 0, static_cast<int64_t>(z)},
                           CuboidShape{imageShape.width, imageShape.height, zEnd - z}};
        fn(reader.getSubimage(box));
        // Streaming: drop the chunk's cached buffer (padding overshoot keeps
        // memory reserved in the reader), so the next chunk can be read under
        // the same cap.
        reader.clearCache();
    }
}

} // namespace

float estimateImageBackground(ImageReader& reader,
                              const CuboidShape& imageShape,
                              const CuboidShape& chunkHint,
                              size_t readerThreads) {
    const BackgroundStatistics stats = computeBackgroundStatistics(
        [&](auto&& fn) {
            forEachImageChunk(reader, imageShape, chunkHint, readerThreads, fn);
        });
    if (stats.count() == 0) return 0.0f;

    const float bg = stats.background();
    spdlog::get("deconvolution")->debug(
        "Estimated whole-image background {:.6f} (mode below the {:.0f}th percentile of {} voxels, range [{:.6f}, {:.6f}])",
        bg, BackgroundStatistics::kSearchQuantile * 100.0f, stats.count(), stats.min(), stats.max());
    return bg;
}

float estimateBackgroundFromSampleCube(const ReaderHandler& reader,
                                       const BoxCoordWithPadding& cube) {
    // Unpadded read of the cube box: padding fill never enters the estimate.
    const Image3D chunk = reader.getSubimage(cube.box);

    const BackgroundStatistics stats = computeBackgroundStatistics(
        [&](auto&& fn) { fn(chunk); });
    if (stats.count() == 0) {
        throw std::runtime_error("No voxels counted from the background sample cube");
    }

    const float bg = stats.background();
    spdlog::get("deconvolution")->debug(
        "Background from sample cube ({}): {:.6f} (mode below the {:.0f}th percentile, range [{:.6f}, {:.6f}])",
        cube.box.print(),
        bg, BackgroundStatistics::kSearchQuantile * 100.0f, stats.min(), stats.max());
    return bg;
}
