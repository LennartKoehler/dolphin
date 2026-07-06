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
#include "dolphinbackend/IComputeBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include "dolphinbackend/IBackendManager.h"
#include "cuda_backend/CUDABackend.h"
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
class CUDAComputeBackend;
class CUDABackend;


/**
 * CUDABackendManager manages thread-specific CUDA backends directly.
 * This singleton class ensures each host thread gets its own CUDA backend,
 * providing clear ownership semantics and lifetime management.
 */
class CUDABackendManager : public IBackendManager{
public:

    explicit CUDABackendManager();
    ~CUDABackendManager() override = default;
    void init(LogCallback fn) override;

    // virtual IComputeBackend& getComputeBackend(const BackendConfig& config) override;
    // virtual IBackendMemoryManager& getBackendMemoryManager(const BackendConfig& config) override;
    virtual IBackend& createBackendForCurrentThread(const BackendConfig& config) override;


    // IBackend& clone(IBackend& backend, const BackendConfig& config) override ;
    IBackend& createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config) override;


    void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) override;

    int getNumberDevices() const override;
protected:

    CUDABackendConfig config;
    std::vector<CUDADevice> devices;

    std::vector<std::unique_ptr<CUDABackend>> backends;
    std::vector<std::unique_ptr<CUDAComputeBackend>> computeBackends;
    std::vector<std::unique_ptr<CUDABackendMemoryManager>> memoryManagers;

    // Threading synchronization

    LogCallback logger_;
    std::mutex mutex_;
    // Configuration
    int nDevices = 0;
    int usedDeviceCounter = 0;

    CUDABackendConfig configToConfig(const BackendConfig& config);
    CUDABackend& createNewBackend(CUDABackendConfig config);
    cudaStream_t createStream() const ;

    virtual std::unique_ptr<CUDAComputeBackend> createComputeBackend(CUDABackendConfig config);
    virtual std::unique_ptr<CUDABackendMemoryManager> createMemoryManager(CUDABackendConfig config);
};
