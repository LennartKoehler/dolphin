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
#include "dolphinbackend/IDeconvolutionBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include "dolphinbackend/IBackendManager.h"
#include "CUDABackend.h"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <array>
#include <thread>
#include <unordered_map>
#include <cuda_runtime.h>

// Forward declarations
class CUDABackendMemoryManager;
class CUDADeconvolutionBackend;
class CUDABackend;


/**
 * CUDABackendManager manages thread-specific CUDA backends directly.
 * This singleton class ensures each host thread gets its own CUDA backend,
 * providing clear ownership semantics and lifetime management.
 */
class CUDABackendManager : public IBackendManager{
public:

    explicit CUDABackendManager(CUDABackendConfig config);
    ~CUDABackendManager() override = default;
    void setLogger(LogCallback fn) override;

    IDeconvolutionBackend* getDeconvolutionBackend(const BackendConfig& config) override;
    IBackendMemoryManager* getBackendMemoryManager(const BackendConfig& config) override;
    IBackend* getBackend(const BackendConfig& config) override;


    IBackend& clone(IBackend& backend, const BackendConfig& config) override ;
    IBackend& cloneSharedMemory(IBackend& backend, const BackendConfig& config) override;


    void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) override;

    int getNumberDevices() const override;
private:

    CUDABackendConfig config;
    std::vector<CUDADevice> devices;

    std::vector<std::shared_ptr<CUDABackend>> backends;
    std::vector<std::unique_ptr<CUDADeconvolutionBackend>> deconvBackends;
    std::vector<std::unique_ptr<CUDABackendMemoryManager>> memoryManagers;

    // Threading synchronization

    LogCallback logger_;
    std::mutex mutex_; 
    // Configuration
    int nDevices;
    int usedDeviceCounter = 0;
    
    // Helper methods
    std::shared_ptr<CUDABackend> createNewBackend(CUDADevice device);
    cudaStream_t createStream();
};
