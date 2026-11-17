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

#include "CuboidShape.h"
#include <string>

/**
 * @brief Memory a system will hold at once, split by physical pool.
 *
 * hostBytes are drawn from the host RAM pool, deviceBytes from the compute
 * device pool (GPU VRAM). For backends where device == host (CPU) the
 * consumer of the requirement merges both into the single host pool.
 */
struct MemoryRequirement {
    size_t hostBytes = 0;
    size_t deviceBytes = 0;

    MemoryRequirement operator+(const MemoryRequirement& other) const {
        return MemoryRequirement{hostBytes + other.hostBytes, deviceBytes + other.deviceBytes};
    }
    MemoryRequirement& operator+=(const MemoryRequirement& other) {
        hostBytes += other.hostBytes;
        deviceBytes += other.deviceBytes;
        return *this;
    }
};

/**
 * @brief Declares how much memory an owning system needs for a given cube.
 *
 * Every class that allocates memory owns a MemoryComponent (created in its
 * constructor, configured via its getMemoryComponent(...) getter). The
 * footprint must report the PEAK held during the processing of one cube of
 * the given (padded) shape, so image splitting can check candidate sizes
 * against the real budget instead of precomputed constants.
 *
 * Contract: footprint() must be pure and cheap (called once per candidate
 * shape in the splitter's search loop) and must only use information the
 * owner already has - it is the single source of truth for that owner's
 * memory, so keep it next to the allocation code it describes.
 */
class MemoryComponent {
public:
    virtual ~MemoryComponent() = default;

    virtual MemoryRequirement footprint(const CuboidShape& cubeShape) const = 0;

    /**
     * @brief Apply this system's runtime cap, so its own tracking enforces
     * what the footprint promised.
     *
     * Called by the strategy after sizing, before the first allocation. The
     * requirement carries the cap per pool; a system that can only cap one
     * pool applies that plane. Default: nothing persistent to cap (transient
     * allocations are covered by the owner pools).
     */
    virtual void setMemoryLimit(const MemoryRequirement& limit) { (void)limit; }

    // Name used in breakdowns and logs (e.g. "reader", "fft workspace").
    virtual std::string name() const = 0;
};
