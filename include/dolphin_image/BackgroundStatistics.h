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

#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin_image/Types/PaddingFillType.h"

// Streaming background estimate for a whole image.
//
// The background (camera offset + constant stray light) is a property of the
// image, not of a subimage. Estimating it per subimage makes neighbouring cubes
// disagree and leaves visible tile seams. BackgroundStatistics therefore
// accumulates ONE histogram over the entire image, chunk by chunk, so the image
// is never held in RAM; it reports the mean of the dimmest decile.
//
// Only voxels that come from the image itself are counted: a chunk read with
// outside-image padding has the padded border excluded, so the padding fill
// (which fades towards zero, see ImagePadding::padImageLinear) does not bias the
// estimate. With a zero fill the border IS counted (zeros are the true
// background there).
//
// Two streaming passes over the counted voxels:
//   1. addForRange: global min/max and count (no memory)
//   2. setRange + addForHistogram: a fixed log-histogram, then background()
// Callers feed both passes through forEachCountedVoxel, e.g. by re-reading the
// image in chunks.
class BackgroundStatistics {
public:
    // Fraction of the dimmest voxels that is averaged (0.1 = dimmest decile).
    static constexpr float kDarkFraction = 0.1f;

    // Values below this are clamped into the first bin (relative bins cannot
    // resolve down to 0).
    static constexpr float kMinQuantile = 1e-12f;

    // Log bins (~1% relative resolution) between min and max.
    static constexpr size_t kNumBins = 1024;
    // The bins span [min, min * kRangeSpread]; anything above is clamped into
    // the last bin. The true max is unsuitable as an edge: a few bright voxels
    // would collapse all background voxels into the first bin.
    static constexpr double kRangeSpread = 10.0;

    BackgroundStatistics() = default;

    // Pass 1: grow the value range and count. Ignores non-finite samples.
    void addForRange(float v);

    // Called between the passes, with the range observed in pass 1.
    void setRange(float min, float max);

    // Pass 2: accumulate the histogram.
    void addForHistogram(float v);

    // Mean of the dimmest kDarkFraction of all counted voxels (0 if empty).
    float background() const;

    size_t count() const { return count_; }
    float min() const { return min_; }
    float max() const { return max_; }

private:
    double binLower(size_t i) const;
    double binUpper(size_t i) const;
    size_t binOf(double v) const;

    float min_ = 0.0f;
    float max_ = 0.0f;
    size_t count_ = 0;

    double binLo_ = 0.0;
    double binHi_ = 0.0;
    size_t inHistogram_ = 0;
    bool degenerate_ = false;
    double bins_[kNumBins] = {};
};

// Calls fn(value) for every voxel of `image` that comes from the image itself
// (see the counting rule above). This is the canonical definition of "counted
// voxel"; both BackgroundStatistics passes must be fed through it.
template <typename Fn>
void forEachCountedVoxel(const Image3D& image, const Padding& padding, PaddingFillType fill, Fn&& fn) {
    const CuboidShape shape = image.getShape();
    size_t x0 = 0, y0 = 0, z0 = 0;
    size_t x1 = shape.width, y1 = shape.height, z1 = shape.depth;
    if (fill != PaddingFillType::ZERO) {
        // A real fill fades the border towards zero: exclude it.
        x0 = std::min<size_t>(padding.before.width, shape.width);
        y0 = std::min<size_t>(padding.before.height, shape.height);
        z0 = std::min<size_t>(padding.before.depth, shape.depth);
        x1 = shape.width - std::min<size_t>(padding.after.width, shape.width - x0);
        y1 = shape.height - std::min<size_t>(padding.after.height, shape.height - y0);
        z1 = shape.depth - std::min<size_t>(padding.after.depth, shape.depth - z0);
    }
    for (size_t z = z0; z < z1; ++z) {
        for (size_t y = y0; y < y1; ++y) {
            for (size_t x = x0; x < x1; ++x) {
                fn(image.getPixel(x, y, z));
            }
        }
    }
}

namespace BackgroundStatisticsDetail {

// Pass 1 callback for forEachCountedVoxel.
struct RangeAccumulator {
    BackgroundStatistics& stats;
    void operator()(float v) const { stats.addForRange(v); }
};

// Pass 2 callback for forEachCountedVoxel.
struct HistogramAccumulator {
    BackgroundStatistics& stats;
    void operator()(float v) const { stats.addForHistogram(v); }
};

} // namespace BackgroundStatisticsDetail
