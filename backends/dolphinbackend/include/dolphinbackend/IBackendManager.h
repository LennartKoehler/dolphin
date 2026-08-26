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




// manages all backends of its type, also is responsible for lifetime of these
class IBackendManager{
public:
    IBackendManager() = default;
    virtual ~IBackendManager() = default;


    virtual void init(LogCallback fn) = 0;

    virtual IBackend& createBackendForCurrentThread(const BackendConfig& config) = 0;

    virtual IBackend& createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config) = 0;

    virtual int getNumberDevices() const = 0;

    // input is the number of threads given through the config
    // the backend may decide that for example it uses omp backends,
    // so it will set the number of omp threads to workerThreads and set workerThreads to 1
    // the returned number of ioThreads and workerThreads is the number of actual std::threads used in the threadpool
    // of the deconvolution executor
    // the backendconfigs is what this manager will later recieve to init backends
    virtual void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) = 0;
};
