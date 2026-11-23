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

// Runs one streaming pass over all z-chunks of the image. `fn(image)` is
// called per chunk with the unpadded chunk (only real image voxels).
template <typename Fn>
void forEachChunk(ImageReader& reader, const CuboidShape& imageShape,
                  const CuboidShape& chunkHint, size_t threads, Fn&& fn) {
    const size_t depth = chunkDepth(imageShape.depth, chunkHint, threads);

    ReaderHandler handler(std::shared_ptr<ImageReader>(&reader, [](ImageReader*){}), PaddingFillType::ZERO);

    for (size_t z = 0; z < imageShape.depth; z += depth) {
        const size_t zEnd = std::min(z + depth, imageShape.depth);
        // Unpadded read: every voxel of the chunk is real image content, so
        // no synthetic fill can bias the estimate.
        BoxCoord box{CuboidPosition{0, 0, static_cast<int64_t>(z)},
                     CuboidShape{imageShape.width, imageShape.height, zEnd - z}};
        Image3D chunk = handler.getSubimage(box);
        fn(chunk);
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
    BackgroundStatistics stats;

    // Pass 1: value range (and count).
    forEachChunk(reader, imageShape, chunkHint, readerThreads,
                 [&stats](const Image3D& chunk) {
                     forEachCountedVoxel(chunk,
                                         BackgroundStatisticsDetail::RangeAccumulator{stats});
                 });
    if (stats.count() == 0) return 0.0f;

    stats.setRange(stats.min(), stats.max());

    // Pass 2: histogram over the counted voxels.
    forEachChunk(reader, imageShape, chunkHint, readerThreads,
                 [&stats](const Image3D& chunk) {
                     forEachCountedVoxel(chunk,
                                         BackgroundStatisticsDetail::HistogramAccumulator{stats});
                 });

    const float bg = stats.background();
    spdlog::get("deconvolution")->debug(
        "Estimated whole-image background {:.6f} (mode below the {:.0f}th percentile of {} voxels, range [{:.6f}, {:.6f}])",
        bg, BackgroundStatistics::kSearchQuantile * 100.0f, stats.count(), stats.min(), stats.max());
    return bg;
}
