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

#include <string>

enum class ThreadMode { Auto, Total, Full, Invalid };

// The user's thread intent, as configured through the frontend.
// The mode is derived from the values (never stored): Auto = nothing set,
// Total = n_threads alone, Full = worker + io + reader pools set.
// Backends resolve this intent into concrete pool sizes and per-op
// parallelism (see IBackendManager::setThreadDistribution).
struct ThreadConfig {
    size_t nThreads        = 0;
    size_t nWorkerThreads  = 0;
    size_t nIOThreads      = 0;
    size_t nReaderThreads  = 0;

    ThreadMode getMode() const;
    bool validate(std::string& error) const;
    std::string describe() const;
    size_t resolvePSFThreadCount() const;
    size_t resolveReaderThreads() const;
};
