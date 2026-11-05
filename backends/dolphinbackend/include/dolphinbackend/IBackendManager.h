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

    virtual IBackend& createBackendForCurrentThread(const BackendConfig& config) = 0;

    virtual IBackend& createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config) = 0;

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
};
