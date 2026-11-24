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
#include <optional>
#include <utility>

#include "dolphin_image/Image3D.h"
#include "dolphin_image/Types/BoxCoord.h"
#include "dolphin_image/Types/PaddingFillType.h"

// Streaming background estimate for a whole image.
//
// The background (camera offset + constant stray light) is a property of the
// image, not of a subimage. Estimating it per subimage makes neighbouring cubes
// disagree and leaves visible tile seams. BackgroundStatistics therefore
// accumulates ONE histogram over the entire image, chunk by chunk, so the image
// is never held in RAM.
//
// The estimate is the MODE of the dark cluster: the percentile
// (kSearchQuantile) only bounds the search, because the background may occupy
// a small minority of the voxels (dense objects/haze) - every percentile below
// that fraction lands in the foreground. The mode search finds the darkest
// substantial, prominent cluster below the bound instead.
//
// Two streaming passes over the counted voxels:
//   1. addForRange: global min/max and count (no memory)
//   2. setRange + addForHistogram: a log-histogram over the FULL range, then
//      background() locates the dark cluster's mode in it.
// Callers feed both passes through forEachCountedVoxel, e.g. by re-reading the
// image in chunks.
class BackgroundStatistics {
public:
    // Fraction of all counted voxels that bounds the mode search (and serves
    // as the fallback percentile estimate when no usable dark cluster exists
    // below it).
    static constexpr float kSearchQuantile = 0.05f;

    // A dark cluster only counts as the background candidate if its
    // half-maximum window holds at least this fraction of all counted voxels.
    // Isolated spikes (repeated dead/hot pixel values) are far below it.
    static constexpr float kMinClusterFraction = 0.01f;

    // Floor of the log bins: relative bins cannot resolve down to 0, and a
    // degenerate anchor (min = 0) would collapse the whole range.
    static constexpr float kMinQuantile = 1e-12f;

    // Log bins over the FULL [floor, max] range of the data. The wide range is
    // what makes the estimate robust: a dark outlier (zero or positive) below
    // the background can sit in the low bins without pulling the percentile,
    // and the background cluster is never clamped into the top bin.
    static constexpr size_t kNumBins = 4096;

    BackgroundStatistics() = default;

    // Pass 1: grow the value range and count. Ignores non-finite samples.
    void addForRange(float v);

    // Called between the passes, with the range observed in pass 1.
    void setRange(float min, float max);

    // Pass 2: accumulate the histogram.
    void addForHistogram(float v);

    // The background level: mode of the dark cluster below the
    // kSearchQuantile bound (0 if empty).
    float background() const;

    size_t count() const { return count_; }
    float min() const { return min_; }
    float max() const { return max_; }

private:
    // Result of the quantile walk: the bin holding the rank and the fraction
    // of it at which the rank falls (uniform-in-log assumption).
    struct QuantileRank {
        size_t bound;
        double frac;
    };

    QuantileRank quantileRank(double target) const;
    double binValueAt(double index) const;
    std::optional<size_t> darkClusterPeak(size_t bound, double total) const;
    std::pair<size_t, size_t> halfMaxWindow(size_t peak) const;
    double windowMean(size_t lo, size_t hi) const;
    double prominence(size_t peak, size_t bound) const;

    double binLower(size_t i) const;
    double binUpper(size_t i) const;
    size_t binOf(double v) const;

    float min_ = 0.0f;
    float max_ = 0.0f;
    size_t count_ = 0;

    // Pass-1 state used to anchor the histogram. Samples <= 0 (exact zeros,
    // dead pixels) cannot live on log bins; they form their own bin with
    // representative value 0 in the histogram pass. The histogram floor is the
    // smallest strictly positive sample: anchoring on min_ instead would let a
    // handful of zero voxels collapse the entire range (every real value
    // clamps into the last bin, whose representative is ~kMinQuantile).
    bool hasNonzero_ = false;
    float minNonzero_ = 0.0f;

    double binLo_ = 0.0;
    double binHi_ = 0.0;
    size_t inHistogram_ = 0;
    size_t zeroBinCount_ = 0;
    bool degenerate_ = false;
    double bins_[kNumBins] = {};
};

// Calls fn(value) for every voxel of `image`. Both BackgroundStatistics passes
// must see the same voxels, so this is the canonical iteration to feed them
// with. Data that must not enter the estimate (e.g. synthetic padding fill) is
// kept out by the read layer, not here.
template <typename Fn>
void forEachCountedVoxel(const Image3D& image, Fn&& fn) {
    const CuboidShape shape = image.getShape();
    for (size_t z = 0; z < shape.depth; ++z) {
        for (size_t y = 0; y < shape.height; ++y) {
            for (size_t x = 0; x < shape.width; ++x) {
                fn(image.getPixel(x, y, z));
            }
        }
    }
}

// Runs the complete two-pass estimate over a data source: `forEachChunk` calls
// fn(chunk) once per chunk of image data, and is invoked once per pass, so the
// source must be re-readable (e.g. re-reading the image from disk in chunks).
// Returns the finished statistics; call background() on the result.
template <typename ForEachChunk>
BackgroundStatistics computeBackgroundStatistics(ForEachChunk&& forEachChunk) {
    BackgroundStatistics stats;
    // Pass 1: value range (and count).
    forEachChunk([&stats](const Image3D& chunk) {
        forEachCountedVoxel(chunk,
                            [&stats](float v) { stats.addForRange(v); });
    });
    // An empty source stays empty; no second pass can add anything.
    if (stats.count() == 0) return stats;
    stats.setRange(stats.min(), stats.max());
    // Pass 2: histogram over the same voxels.
    forEachChunk([&stats](const Image3D& chunk) {
        forEachCountedVoxel(chunk,
                            [&stats](float v) { stats.addForHistogram(v); });
    });
    return stats;
}
