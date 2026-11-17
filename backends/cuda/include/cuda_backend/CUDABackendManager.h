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
 * @brief MemoryComponent for the CUDA manager's cuFFT work areas.
 *
 * Every backend instance creates an r2c and a c2r cuFFT plan per cube shape
 * (the pipeline always passes distinct in/out buffers, so no in-place
 * variants); with cuFFT's default auto-allocation each cached plan keeps its
 * work area allocated for the plan's lifetime, per instance.
 * cufftEstimate3d queries the work-area size without creating a plan (pure
 * host-side arithmetic, no CUDA context required). Reports and caps the
 * DEVICE plane only.
 */
class CUDAFFTWorkspaceComponent : public MemoryComponent {
public:
    explicit CUDAFFTWorkspaceComponent(CUDABackendManager& manager) : manager(manager) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape) const override {
        size_t workR2C = 0;
        size_t workC2R = 0;
        if (cufftEstimate3d(cubeShape.depth, cubeShape.height, cubeShape.width, CUFFT_R2C, &workR2C) != CUFFT_SUCCESS
            || cufftEstimate3d(cubeShape.depth, cubeShape.height, cubeShape.width, CUFFT_C2R, &workC2R) != CUFFT_SUCCESS) {
            // Conservative fallback matching the old heuristic (2 cube volumes).
            return MemoryRequirement{0, 2 * cubeShape.getVolume() * sizeof(real_t)};
        }
        return MemoryRequirement{0, workR2C + workC2R};
    }

    // Facade for the manager's per-device VRAM cap.
    void setMemoryLimit(const MemoryRequirement& limit) override;

    std::string name() const override { return "fft workspace"; }

private:
    CUDABackendManager& manager;
};


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

    IBackend& createBackendForCurrentThread(const BackendConfig& config) override;


    // IBackend& clone(IBackend& backend, const BackendConfig& config) override ;
    IBackend& createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config) override;


    void setThreadDistribution(ThreadConfig& inputThreadConfig, BackendConfig& ioconfig, BackendConfig& workerConfig) override;

    size_t getNumberDevices() const override;

    // The GPU does the heavy work; a handful of host threads is enough to keep
    // every device fed with reads, copies and writes.
    size_t getMaxNumberThreads() const override;

    // Free VRAM on the least capable managed device, queried through the
    // shared CUDABackendMemoryManager::staticGetAvailableMemory primitive.
    size_t getAvailableMemoryPerDevice() const override;
    // Applies the same allocation cap to every managed device (via each
    // device's MemoryTracking, the same state its memory manager enforces).
    void setMemoryLimit(size_t maxMemoryPerDevice_byte) override;

    MemoryComponent& getMemoryComponent() override;
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

    CUDAFFTWorkspaceComponent fftComponent;

    CUDABackendConfig configToConfig(const BackendConfig& config);
    CUDABackend& createNewBackend(CUDABackendConfig config);
    cudaStream_t createStream() const ;

    virtual std::unique_ptr<CUDAComputeBackend> createComputeBackend(CUDABackendConfig config);
    virtual std::unique_ptr<CUDABackendMemoryManager> createMemoryManager(CUDABackendConfig config);
};

// Defined after CUDABackendManager (it calls into it).
inline void CUDAFFTWorkspaceComponent::setMemoryLimit(const MemoryRequirement& limit) {
    manager.setMemoryLimit(limit.deviceBytes);
}
