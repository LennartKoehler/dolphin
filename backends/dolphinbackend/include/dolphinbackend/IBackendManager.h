#pragma once

#include "IBackendMemoryManager.h"
#include "IDeconvolutionBackend.h"
#include "IBackend.h"




// manages all backends of its type, also is responsible for lifetime of these
class IBackendManager{
public: 
    IBackendManager() = default;
    virtual ~IBackendManager() = default;


    virtual void setLogger(LogCallback fn) = 0;

    virtual IDeconvolutionBackend& getDeconvolutionBackend(const BackendConfig& config) = 0;
    virtual IBackendMemoryManager& getBackendMemoryManager(const BackendConfig& config) = 0;
    virtual IBackend& getBackend(const BackendConfig& config) = 0;

    virtual IBackend& clone(IBackend& backend, const BackendConfig& config) = 0;
    virtual IBackend& cloneSharedMemory(IBackend& backend, const BackendConfig& config) = 0;

    virtual int getNumberDevices() const = 0;

    // input is the number of threads given through the config
    // the backend may decide that for example it uses omp backends,
    // so it will set the number of omp threads to workerThreads and set workerThreads to 1
    // the returned number of ioThreads and workerThreads is the number of actual std::threads used in the threadpool
    // of the deconvolution executor
    // the backendconfigs is what this manager will later recieve to init backends
    virtual void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) = 0;
};