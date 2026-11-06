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
#include "MemoryComponent.h"
#include <memory>



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

    // FFT workspace estimate for a candidate cube, in units of one cube volume
    // of real_t. Both current backends use a constant ~2 copies; override when
    // a backend's FFT library has different (e.g. shape-dependent) overhead.
    virtual float estimateFFTWorkspaceCopies(const CuboidShape& shape) const {
        return 2;
    }

    // MemoryComponent describing this backend's FFT workspace demand. The
    // component is owned by the manager and configured on first call.
    MemoryComponent& getMemoryComponent();

private:
    std::unique_ptr<MemoryComponent> memoryComponent;
};

/**
 * @brief MemoryComponent for the backend manager's memory.
 *
 * Lives on the backend manager because the FFT workspace depends on the
 * backend's FFT library (cuFFT/FFTW) and must be queryable before any backend
 * is constructed. Reports the workspace in units of one cube volume of real_t.
 * Also acts as the facade for the manager's runtime allocation cap
 * (setMemoryLimit): accelerator managers cap per-device VRAM (device plane),
 * the CPU manager caps its shared host-RAM tracker (host plane).
 * Defined after IBackendManager (it calls into it).
 */
class FFTWorkspaceComponent : public MemoryComponent {
public:
    explicit FFTWorkspaceComponent(IBackendManager& manager) : manager(manager) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape) const override {
        const float copies = manager.estimateFFTWorkspaceCopies(cubeShape);
        return MemoryRequirement{0, static_cast<size_t>(copies * cubeShape.getVolume() * sizeof(real_t))};
    }

    void setMemoryLimit(const MemoryRequirement& limit) override {
        // The strategy passes exactly the plane this manager owns.
        manager.setMemoryLimit(limit.deviceBytes > 0 ? limit.deviceBytes : limit.hostBytes);
    }

    std::string name() const override { return "fft workspace"; }

private:
    IBackendManager& manager;
};

inline MemoryComponent& IBackendManager::getMemoryComponent() {
    if (!memoryComponent) {
        memoryComponent = std::make_unique<FFTWorkspaceComponent>(*this);
    }
    return *memoryComponent;
}
