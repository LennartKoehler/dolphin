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

#include "MemoryComponent.h"
#include <vector>

/**
 * @brief Composite of all MemoryComponents of one deconvolution run.
 *
 * The strategy collects the MemoryComponent of every memory-owning system
 * (reader, io staging, algorithm, fft workspace, psf cache, padding) into a
 * footprint and passes it to the image splitter, which asks for the total
 * demand of each candidate cube shape and compares it against the resolved
 * MemoryBudget. The components are borrowed (not owned) and must outlive the
 * footprint.
 *
 * Each component reports the memory of ONE instance; the strategy applies the
 * scheduling (how many instances run concurrently) as a count at add-time.
 */
class MemoryFootprint {
public:
    MemoryFootprint() = default;

    // Borrows a component, held count times simultaneously (default 1).
    // count is the number of concurrent instances of this system (e.g. one
    // algorithm working set per worker thread, one io staging set per io
    // thread) - the scheduling decision lives here, not in the components.
    void add(const MemoryComponent& component, size_t count = 1);

    // Summed demand of all components for one candidate cube shape.
    MemoryRequirement total(const CuboidShape& cubeShape) const;

    // One line per component plus the total - for logs and error messages
    // that say which system binds the budget.
    std::string breakdown(const CuboidShape& cubeShape) const;

private:
    struct Entry {
        const MemoryComponent* component;
        size_t count;
    };
    std::vector<Entry> entries;
};
