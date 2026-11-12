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

#include "dolphinbackend/MemoryFootprint.h"

#include <sstream>
#include <iomanip>

void MemoryFootprint::add(const MemoryComponent& component, size_t count) {
    entries.push_back(Entry{&component, count, nullptr});
}

void MemoryFootprint::add(std::unique_ptr<const MemoryComponent> component, size_t count) {
    entries.push_back(Entry{component.get(), count, std::move(component)});
}

MemoryRequirement MemoryFootprint::total(const CuboidShape& cubeShape) const {
    MemoryRequirement sum;
    for (const Entry& entry : entries) {
        MemoryRequirement need = entry.component->footprint(cubeShape);
        need.hostBytes *= entry.count;
        need.deviceBytes *= entry.count;
        sum += need;
    }
    return sum;
}

std::string MemoryFootprint::breakdown(const CuboidShape& cubeShape) const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    MemoryRequirement sum;
    for (const Entry& entry : entries) {
        MemoryRequirement need = entry.component->footprint(cubeShape);
        need.hostBytes *= entry.count;
        need.deviceBytes *= entry.count;
        sum += need;
        out << entry.component->name() << (entry.count > 1 ? " (x" + std::to_string(entry.count) + ")" : "")
            << ": " << need.hostBytes / 1073741824.0 << " GiB host, "
            << need.deviceBytes / 1073741824.0 << " GiB device; ";
    }
    out << "total: " << sum.hostBytes / 1073741824.0 << " GiB host, "
        << sum.deviceBytes / 1073741824.0 << " GiB device";
    return out.str();
}
