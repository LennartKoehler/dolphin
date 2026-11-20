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
    ++count_;
}

void BackgroundStatistics::setRange(float min, float max) {
    // Relative (log) bins need a strictly positive, non-degenerate interval.
    // Remember whether the data is degenerate: then the exact value is known.
    degenerate_ = (max <= min);
    double lo = std::max<double>(static_cast<double>(min), kMinQuantile);
    double hi = std::min<double>(static_cast<double>(max), lo * (1.0 + kRangeSpread));
    if (hi <= lo) hi = lo * (1.0 + kRangeSpread);
    binLo_ = lo;
    binHi_ = hi;
    std::fill(std::begin(bins_), std::end(bins_), 0.0);
    inHistogram_ = 0;
}

void BackgroundStatistics::addForHistogram(float v) {
    if (!std::isfinite(v) || binHi_ <= binLo_) return;
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
    if (inHistogram_ == 0) return min_;

    // Average bins, bottom-up, until the dimmest fraction is covered. Each bin
    // is assumed uniform between its edges, so its mass is split proportionally
    // when the target falls inside a bin (error below the ~1% bin resolution).
    const double target = static_cast<double>(inHistogram_) * kDarkFraction;
    double acc = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < kNumBins && acc < target; ++i) {
        const double take = std::min<double>(static_cast<double>(bins_[i]), target - acc);
        if (take <= 0.0) continue;
        sum += 0.5 * (binLower(i) + binUpper(i)) * take;
        acc += take;
    }
    if (acc <= 0.0) return min_;
    return static_cast<float>(sum / acc);
}