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

#include "IBackendMemoryManager.h"
#include "IComputeBackend.h"
#include "IBackend.h"
#include "ThreadConfig.h"



// manages all backends of its type, also is responsible for lifetime of these
class IBackendManager{
public:
    IBackendManager() = default;
    virtual ~IBackendManager() = default;


    virtual void init(LogCallback fn) = 0;

    virtual IBackend& createBackend(const BackendConfig& config) = 0;

    virtual size_t getNumberDevices() const = 0;

    // Upper bound on useful hardware threads for this backend. Used to resolve
    // "auto" (0) thread counts. Device backends usually return a small number
    // since the host only needs to keep the device fed.
    virtual size_t getMaxNumberThreads() const { return 1; }

    // The backend owns all thread configuration. It receives the user's thread
    // intent (Auto / Total / Full, derived from the values) and resolves it:
    //  - writes the concrete executor pool sizes back into threadConfig
    //    (nIOThreads / nWorkerThreads = the number of std::threads in the io /
    //    worker pool the deconvolution executor runs, nReaderThreads = the
    //    reader pool size), and
    //  - fills ioconfig / workerConfig with the per-op parallelism that backends
    //    of this type use to run a task (e.g. OMP threads per worker op), which
    //    the strategy later hands back to this manager to init backends.
    // The strategy does no thread math of its own; it just reads the resolved
    // values back.
    virtual void setThreadDistribution(ThreadConfig& inputThreadConfig, BackendConfig& ioconfig, BackendConfig& workerConfig) = 0;

    // --- Memory capabilities and configuration (mirrors the thread pattern) ---
    //
    // The per-device memory managers (IBackendMemoryManager) own the actual
    // tracking and enforcement. These manager-level methods exist because the
    // strategy must query and configure memory BEFORE any backend exists.
    // Implementations must delegate to the same primitives/state the memory
    // managers use (shared statics, per-device MemoryTrackings), never
    // re-implement backend logic.

    // Memory currently available (free) on the least capable device this
    // manager drives, without constructing any backend. The CPU backend
    // reports free host RAM; CUDA reports the smallest free VRAM across its
    // devices. Used to resolve "auto" memory budgets.
    virtual size_t getAvailableMemoryPerDevice() const {
        NOT_IMPLEMENTED(getAvailableMemoryPerDevice);
    }

    // Apply a hard per-device allocation cap. Allocations beyond the cap fail
    // with a clear MemoryException instead of exhausting the machine. Must be
    // called before the first backend of the run allocates.
    virtual void setMemoryLimit(size_t maxMemoryPerDevice_byte) {
        NOT_IMPLEMENTED(setMemoryLimit);
    }

    // --- FFT plan memory, per plan set (mirrors the
    // getAvailableMemoryPerDevice pattern: the deconvolution strategy must
    // model memory BEFORE any backend exists, so the query lives on the
    // manager and is answered with the compute backend's plan math) ---
    //
    // The algorithms' memory components hand over the FFTPlanDescriptions
    // their algorithm will use (e.g. PoissonMLECore::planDescriptions) plus
    // the run's worker thread count; the answer is composed into the
    // algorithm's memory component and never added to the footprint directly.

    // Memory the given plan set will occupy on this backend's plane, with
    // this backend's sharing AND thread policy applied inside: CUDA creates
    // one shared work area per backend instance (max over the set, times the
    // worker backends); the CPU's FFTW planning holds no persistent memory,
    // so it answers 0. Estimated WITHOUT creating any plan (pure host-side
    // arithmetic).
    virtual size_t getFFTPlanMemoryBytes(const std::vector<FFTPlanDescription>& descriptions, const size_t workerThreads) const {
        NOT_IMPLEMENTED(getFFTPlanMemoryBytes);
    }

};
