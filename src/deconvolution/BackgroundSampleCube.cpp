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

#include "dolphin/deconvolution/BackgroundSampleCube.h"

#include <algorithm>
#include <spdlog/spdlog.h>

#include "dolphin_image/IO/ReaderWriter.h"

namespace {

BoxCoord imageBox(const ReaderHandler& reader) {
    return BoxCoord{CuboidPosition{0, 0, 0}, reader.getMetaData().getShape()};
}

// Prefer cubes fully inside the image; otherwise the first cube is used and its
// padded border is excluded.
bool isInterior(const BoxCoordWithPadding& cube, const BoxCoord& image) {
    return cube.getPaddedBox().isWithin(image);
}

} // namespace

std::optional<float> estimateBackgroundFromSampleCube(
    const ReaderHandler& reader,
    const std::vector<BoxCoordWithPadding>& cubes) {
    if (cubes.empty()) return std::nullopt;

    const BoxCoord image = imageBox(reader);
    const auto interior = std::find_if(cubes.begin(), cubes.end(),
                                       [&image](const BoxCoordWithPadding& cube) {
                                           return isInterior(cube, image);
                                       });

    const BoxCoordWithPadding& sample = interior != cubes.end() ? *interior : cubes.front();
    const bool fullyInterior = interior != cubes.end();

    PaddedImage chunk = reader.getSubimagePadded(sample);

    BackgroundStatistics stats;
    // Interior cube: all voxels come from the image, count everything. Edge
    // cube: skip the padded border (the fill is synthetic).
    const Padding noPadding{CuboidShape{0, 0, 0}, CuboidShape{0, 0, 0}};
    const Padding& padding = fullyInterior ? noPadding : chunk.padding;
    const PaddingFillType fill = fullyInterior ? PaddingFillType::ZERO : reader.getPaddingFillType();

    forEachCountedVoxel(chunk.image, padding, fill,
                        BackgroundStatisticsDetail::RangeAccumulator{stats});
    if (stats.count() == 0) return std::nullopt;
    stats.setRange(stats.min(), stats.max());
    forEachCountedVoxel(chunk.image, padding, fill,
                        BackgroundStatisticsDetail::HistogramAccumulator{stats});

    const float bg = stats.background();
    spdlog::get("deconvolution")->debug(
        "Background from sample cube ({}){}: {:.6f} (dimmest {:.0f}%, range [{:.6f}, {:.6f}])",
        sample.box.print(), fullyInterior ? ", interior" : ", first cube (border excluded)",
        bg, BackgroundStatistics::kDarkFraction * 100.0f, stats.min(), stats.max());
    return bg;
}