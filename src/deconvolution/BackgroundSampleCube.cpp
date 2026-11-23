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

float estimateBackgroundFromSampleCube(
    const ReaderHandler& reader,
    const BoxCoordWithPadding& cube) {

    const BoxCoord image = imageBox(reader);

    Image3D chunk = reader.getSubimage(cube.box);


    BackgroundStatistics stats;

    forEachCountedVoxel(chunk,
                        BackgroundStatisticsDetail::RangeAccumulator{stats});
    if (stats.count() == 0) throw std::runtime_error("No stats collected from cubes");
    stats.setRange(stats.min(), stats.max());
    forEachCountedVoxel(chunk,
                        BackgroundStatisticsDetail::HistogramAccumulator{stats});

    const float bg = stats.background();
    spdlog::get("deconvolution")->debug(
        "Background from sample cube ({}): {:.6f} (dimmest {:.0f}%, range [{:.6f}, {:.6f}])",
        cube.box.print(),
        bg, BackgroundStatistics::kDarkFraction * 100.0f, stats.min(), stats.max());
    return bg;
}
