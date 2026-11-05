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

#include "dolphinbackend/ThreadConfig.h"

#include <thread>
#include <algorithm>
#include <sstream>

namespace {
size_t hardwareConcurrencySafe() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
}
}

ThreadMode ThreadConfig::getMode() const {
    if (nThreads > 0) {
        // Total mode: n_threads alone, no other field may be set.
        if (nWorkerThreads == 0 && nIOThreads == 0 && nReaderThreads == 0)
            return ThreadMode::Total;
        return ThreadMode::Invalid;
    }
    // n_threads == 0
    if (nReaderThreads > 0 && nWorkerThreads == 0 && nIOThreads == 0)
        return ThreadMode::Invalid;  // reader threads without any worker/io pool
    if (nWorkerThreads > 0 && nIOThreads > 0 && nReaderThreads > 0)
        return ThreadMode::Full;
    return ThreadMode::Auto;
}

bool ThreadConfig::validate(std::string& error) const {
    if (getMode() == ThreadMode::Invalid) {
        error = "Invalid thread configuration (" + describe() + "). "
                "Use exactly one of: "
                "(1) n_threads alone, "
                "(2) n_worker_threads + n_io_threads with optional n_reader_threads, "
                "(3) none (auto). "
                "Do not combine n_threads with n_worker_threads, n_io_threads or n_reader_threads.";
        return false;
    }
    return true;
}

std::string ThreadConfig::describe() const {
    std::ostringstream oss;
    oss << "n_threads=" << nThreads
        << ", n_worker_threads=" << nWorkerThreads
        << ", n_io_threads=" << nIOThreads
        << ", n_reader_threads=" << nReaderThreads;
    return oss.str();
}

size_t ThreadConfig::resolvePSFThreadCount() const {
    if (nThreads > 0) return nThreads;
    if (nWorkerThreads > 0) return nWorkerThreads;
    if (nIOThreads > 0) return nIOThreads;
    return hardwareConcurrencySafe();
}

size_t ThreadConfig::resolveReaderThreads() const {
    if (nReaderThreads > 0) return nReaderThreads;
    // Default: oversubscribed by design (reader pool runs on top of the io pool).
    return std::max(hardwareConcurrencySafe(), nIOThreads);
}
