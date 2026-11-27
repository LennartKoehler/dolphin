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
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <array>
#include <unordered_map>
#include <cuda_runtime.h>
#include <atomic>

// Forward declarations
class CUDABackendMemoryManager;
class CUDAComputeBackend;
class CUDABackend;
class CUDABackendManager;

/**
 * CUDABackendManager manages thread-specific CUDA backends directly.
 * This singleton class ensures each host thread gets its own CUDA backend,
 * providing clear ownership semantics and lifetime management.
 */
class CUDABackendManager : public IBackendManager{
public:

    ~CUDABackendManager() override;
    void init(LogCallback fn) override;

    IBackend& createBackendForCurrentThread(const BackendConfig& config) override;

    void setThreadDistribution(ThreadConfig& inputThreadConfig, BackendConfig& ioconfig, BackendConfig& workerConfig) override;

    size_t getNumberDevices() const override;


    // Free VRAM on the least capable managed device, queried through the
    // shared CUDABackendMemoryManager::staticGetAvailableMemory primitive.
    size_t getAvailableMemoryPerDevice() const override;
    // Applies the same allocation cap to every managed device (via each
    // device's MemoryTracking, the same state its memory manager enforces).
    void setMemoryLimit(size_t maxMemoryPerDevice_byte) override;

    // Max over the plan set's cuFFT estimates: this backend gives every plan
    // of one instance the same shared work area (grown to the largest), so
    // the set costs the largest single plan, not the sum. The estimate is
    // pure host-side arithmetic from the compute backend's plan math.
    size_t getFFTPlanMemoryBytes(const std::vector<FFTPlanDescription>& descriptions, const size_t workerThreads) const override;
protected:

    std::vector<CUDADevice> devices;

    std::deque<std::unique_ptr<CUDABackend>> backends;
    std::deque<std::unique_ptr<CUDAComputeBackend>> computeBackends;
    std::deque<std::unique_ptr<CUDABackendMemoryManager>> memoryManagers;

    // Threading synchronization

    LogCallback logger_;
    std::mutex mutex_;
    // Configuration
    int nDevices = 0;
    std::atomic<int> usedDeviceCounter{0};

    CUDABackend& createNewBackend(CUDABackendConfig config);

    virtual std::unique_ptr<CUDAComputeBackend> createComputeBackend(CUDABackendConfig config);
    virtual std::unique_ptr<CUDABackendMemoryManager> createMemoryManager(CUDABackendConfig config);
};
