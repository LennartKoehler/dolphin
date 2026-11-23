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

#include "dolphin_image/BackgroundStatistics.h"

#include <algorithm>
#include <cmath>

void BackgroundStatistics::addForRange(float v) {
    if (!std::isfinite(v)) return;
    if (count_ == 0) {
        min_ = v;
        max_ = v;
    } else {
        min_ = std::min(min_, v);
        max_ = std::max(max_, v);
    }
    // Remember the smallest strictly positive sample: the histogram floor.
    // A handful of zero voxels (dead pixels) must not drag it to 0.
    if (v > 0.0f && (!hasNonzero_ || v < minNonzero_)) {
        minNonzero_ = v;
        hasNonzero_ = true;
    }
    ++count_;
}

void BackgroundStatistics::setRange(float min, float max) {
    // Log bins need a strictly positive, non-degenerate interval. Remember
    // whether the data is degenerate: then the exact value is known.
    degenerate_ = (max <= min);
    // The floor is the smallest strictly positive sample (kMinQuantile when
    // the data has none): anchoring on min_ would let a handful of zero voxels
    // collapse the whole range. The top edge is the observed max, so the
    // background cluster is always INSIDE the histogram even when dark
    // outliers sit below it (a fixed 1-decade window is what made the old
    // estimate collapse).
    const double lo = hasNonzero_ ? std::max<double>(static_cast<double>(minNonzero_), kMinQuantile)
                                  : static_cast<double>(kMinQuantile);
    const double hi = std::max<double>(static_cast<double>(max), lo * 1.0001);
    binLo_ = lo;
    binHi_ = hi;
    std::fill(std::begin(bins_), std::end(bins_), 0.0);
    inHistogram_ = 0;
    zeroBinCount_ = 0;
}

void BackgroundStatistics::addForHistogram(float v) {
    if (!std::isfinite(v) || binHi_ <= binLo_) return;
    // Samples <= 0 cannot be placed on log bins: they form their own bin with
    // representative value 0.
    if (v <= 0.0f) {
        ++zeroBinCount_;
        return;
    }
    ++bins_[binOf(v)];
    ++inHistogram_;
}

double BackgroundStatistics::binLower(size_t i) const {
    const double t = static_cast<double>(i) / static_cast<double>(kNumBins);
    return binLo_ * std::pow(binHi_ / binLo_, t);
}

double BackgroundStatistics::binUpper(size_t i) const {
    return binLower(i + 1);
}

size_t BackgroundStatistics::binOf(double v) const {
    const double t = std::log(std::max(v, binLo_) / binLo_) / std::log(binHi_ / binLo_);
    return static_cast<size_t>(std::clamp(t, 0.0, 1.0) * static_cast<double>(kNumBins - 1));
}

float BackgroundStatistics::background() const {
    if (count_ == 0) return 0.0f;
    // All data has the same value: that value is the background exactly.
    if (degenerate_) return std::max(min_, 0.0f);
    // No histogram pass was run: the only information is the range.
    const double total = static_cast<double>(inHistogram_ + zeroBinCount_);
    if (total == 0.0) return std::max(min_, 0.0f);

    // Walk to the kSearchQuantile rank. The percentile only BOUNDS the mode
    // search (the background may be a small minority of the voxels, so every
    // percentile below that fraction lands in the foreground) and serves as
    // the fallback estimate when no usable dark cluster exists below it.
    const double target = total * kSearchQuantile;
    // The zero bin alone reaches the rank: the background IS zero.
    if (static_cast<double>(zeroBinCount_) >= target) return 0.0f;

    double acc = static_cast<double>(zeroBinCount_);
    size_t bound = kNumBins - 1;
    double frac = 1.0;
    for (size_t i = 0; i < kNumBins; ++i) {
        const double c = bins_[i];
        if (c > 0.0 && acc + c >= target) {
            bound = i;
            frac = (target - acc) / c;
            break;
        }
        acc += c;
    }
    // Percentile value inside the bound bin (uniform-in-log assumption).
    const auto binValue = [this](double index) {
        const double t = index / static_cast<double>(kNumBins);
        return std::exp(std::log(binLo_) + t * (std::log(binHi_) - std::log(binLo_)));
    };
    const float percentile = static_cast<float>(
        binValue(static_cast<double>(bound) + frac));

    // Half-maximum window around a peak bin: the contiguous run of bins whose
    // count stays at or above half the peak's count. Not capped at the bound:
    // when the bound cuts through the cluster, the window must still span it.
    const auto fwhmRange = [this](size_t peak) {
        const double half = 0.5 * bins_[peak];
        size_t lo = peak;
        while (lo > 0 && bins_[lo - 1] >= half) --lo;
        size_t hi = peak;
        while (hi + 1 < kNumBins && bins_[hi + 1] >= half) ++hi;
        return std::pair<size_t, size_t>{lo, hi};
    };

    // Count-weighted mean over a window, each bin represented by the geometric
    // midpoint of its (log-spaced) edges.
    const auto fwhmMean = [this](size_t lo, size_t hi) {
        double sum = 0.0;
        double mass = 0.0;
        for (size_t i = lo; i <= hi; ++i) {
            sum += bins_[i] * std::sqrt(binLower(i) * binUpper(i));
            mass += bins_[i];
        }
        return mass > 0.0 ? sum / mass : 0.0;
    };

    // Prominence of a peak within [0, bound]: its count minus the highest
    // "col" (the minimum along the terrain) towards higher ground on either
    // side. A real isolated cluster rises far above its surroundings; noise
    // wiggles on a rising flank or a plateau barely stand out.
    const auto prominence = [this, bound](size_t peak) {
        const auto col = [&](int dir) {
            double best = bins_[peak];
            for (int i = static_cast<int>(peak) + dir;
                 i >= 0 && i <= static_cast<int>(bound); i += dir) {
                const double c = bins_[static_cast<size_t>(i)];
                if (c >= bins_[peak]) break;
                best = std::min(best, c);
            }
            return best;
        };
        return bins_[peak] - std::max(col(-1), col(+1));
    };

    // Among the interior local maxima below the bound, keep the most prominent
    // one whose half-maximum window is a substantial cluster. This skips noise
    // wiggles and isolated repeated values (dead/hot pixels) without any
    // absolute threshold; ties keep the darkest candidate.
    size_t chosen = kNumBins;
    double bestProminence = 0.0;
    for (size_t i = 0; i <= bound; ++i) {
        const double c = bins_[i];
        if (c <= 0.0) continue;
        const double left = i > 0 ? bins_[i - 1] : 0.0;
        const double right = i + 1 < kNumBins ? bins_[i + 1] : 0.0;
        if (c < left || c < right || (c == left && c == right)) continue;
        const auto [lo, hi] = fwhmRange(i);
        double mass = 0.0;
        for (size_t j = lo; j <= hi; ++j) mass += bins_[j];
        if (mass < static_cast<double>(kMinClusterFraction) * total) continue;
        const double p = prominence(i);
        if (p > bestProminence) {
            bestProminence = p;
            chosen = i;
        }
    }
    // No substantial dark cluster below the bound (e.g. the bound falls inside
    // a broad unimodal floor): fall back to the percentile itself.
    if (chosen == kNumBins) return percentile;

    const auto [lo, hi] = fwhmRange(chosen);
    return static_cast<float>(fwhmMean(lo, hi));
}