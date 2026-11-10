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
#include "dolphin/ProgressTracking.h"
#include "dolphin/deconvolution/deconvolutionStrategies/IDeconvolutionStrategy.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/MemoryFootprint.h"
#include <functional>
#include <utility>



/// Fraction of the currently available memory used when the user has not set
/// an explicit limit (maxMemHost_gb / maxMemDevice_gb <= 0). Leaves headroom
/// for fragmentation, other processes and model inaccuracies.
static constexpr float AUTO_MEMORY_SAFETY_FACTOR = 0.9f;

/**
 * @brief MemoryComponent for the io pool staging (no owning class exists).
 *
 * Each io thread holds a fixed set of reused scratch buffers (allocated once,
 * reused for every cube): the padded input conversion buffer and the output
 * conversion buffer on the host (2 cube volumes of real_t), plus the input
 * and output device buffers (in-place real layout, ~2 cube volumes of real_t
 * on the device). The footprint applies the number of io threads.
 */
class IOStagingMemoryComponent : public MemoryComponent {
public:
    MemoryRequirement footprint(const CuboidShape& cubeShape) const override {
        const size_t volumeBytes = cubeShape.getVolume() * sizeof(real_t);
        return MemoryRequirement{1 * volumeBytes, 2 * volumeBytes};
    }

    std::string name() const override { return "io staging"; }
};

/**
 * @brief MemoryComponent for the padding transient (no owning class exists).
 *
 * While a cube is padded straight into the io staging input buffer, the
 * unpadded source cube is held alongside it (the padded target IS the staging
 * buffer, accounted by the io staging component).
 */
class PaddingPeakMemoryComponent : public MemoryComponent {
public:
    explicit PaddingPeakMemoryComponent(const Padding& cubePadding) : cubePadding(cubePadding) {}

    MemoryRequirement footprint(const CuboidShape& paddedCube) const override {
        const size_t paddingVolume = cubePadding.getTotalPadding().getVolume();
        const size_t unpaddedVolume = paddedCube.getVolume() > paddingVolume
            ? paddedCube.getVolume() - paddingVolume
            : paddedCube.getVolume();
        const size_t bytes = unpaddedVolume * sizeof(real_t);
        return MemoryRequirement{bytes, 0};
    }

    std::string name() const override { return "padding"; }

private:
    Padding cubePadding;
};

/**
 * @brief MemoryComponent for the algorithm working sets.
 *
 * The algorithm only reports how many bytes its scratch arrays need
 * (DeconvolutionAlgorithm::getWorkingSetBytes); the strategy owns the
 * placement decision: the bytes live on the compute device, or in host RAM
 * when the backend is the CPU (device == host).
 */
class AlgorithmWorkingSetMemoryComponent : public MemoryComponent {
public:
    AlgorithmWorkingSetMemoryComponent(const DeconvolutionAlgorithm& algorithm, bool onDevice)
        : algorithm(algorithm), onDevice(onDevice) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape) const override {
        const size_t bytes = algorithm.getWorkingSetBytes(cubeShape);
        return onDevice ? MemoryRequirement{0, bytes} : MemoryRequirement{bytes, 0};
    }

    std::string name() const override { return "algorithm working set"; }

private:
    const DeconvolutionAlgorithm& algorithm;
    bool onDevice;
};

class StandardDeconvolutionStrategy : public IDeconvolutionStrategy {
public:
    StandardDeconvolutionStrategy() = default;
    ~StandardDeconvolutionStrategy() = default;


    Result<DeconvolutionPlan> createPlan(
        std::shared_ptr<ImageReader> reader,
        std::shared_ptr<ImageWriter> writer,
        PSFHandler& psfHandler,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) override;


protected:

    /// Resolve the raw user limits into byte budgets: -1/0 = auto (use ~90% of
    /// what is currently available), >0 = explicit GB upper bound, always
    /// clamped down to what actually exists. Never raises a limit.
    static size_t resolveMemoryLimit(float limit_gb, size_t available_byte);

    /// How much memory the application thinks one run needs: composes the
    /// footprint of every memory-owning system for the scheduling of this
    /// run. The externally-owned components (reader, writer, backend FFT,
    /// psf cache) are borrowed; the strategy-owned terms (io staging per io
    /// thread, algorithm working set per worker thread with its placement,
    /// padding peak) are created and kept alive by the returned footprint.
    /// Used by createPlan and by tests to inspect the modeled demand.
    static MemoryFootprint buildMemoryFootprint(
        const MemoryComponent& readerComponent,
        const MemoryComponent& writerComponent,
        const MemoryComponent& backendComponent,
        const MemoryComponent& psfComponent,
        const DeconvolutionAlgorithm& algorithm,
        const PaddingScheme& paddingScheme,
        size_t ioThreads,
        size_t workerThreads,
        bool algorithmOnDevice);

    /// Query available memory from the managers and clamp the user limits
    /// (see resolveMemoryLimit). Constructs no backends.
    MemoryBudget resolveMemoryBudget(
        const SetupConfig& config,
        IBackendManager& backendManager) const;

    /// Apply the resolved caps to the backend trackers so allocations beyond
    /// the budget fail with a clear MemoryException. backendComponent is the
    /// memory component of the compute backend (CPU or device).
    void applyBackendMemoryLimits(MemoryComponent& backendComponent, const MemoryBudget& budget) const;

    std::shared_ptr<DeconvolutionAlgorithm> getAlgorithm(const DeconvolutionConfig& config);

    std::vector<BoxCoordWithPadding> getCubes(
        const MemoryBudget& budget,
        const MemoryFootprint& footprint,
        const size_t& minNumberCubes,
        const DeconvolutionConfig& deconvConfig,
        const CuboidShape& imageSize,
        const PaddingScheme& paddingScheme
    ) const;

    std::pair<ThreadConfig, size_t> resolveThreadsAndDevices(
        IBackendManager& manager,
        const size_t& nDevices,
        const ThreadConfig& threadConfig,
        BackendConfig& workerConfig,
        BackendConfig& ioConfig
    ) const;


    virtual std::vector<std::shared_ptr<TaskContext>> createContexts(
        IBackendManager& manager,
        PSFHandler& psfHandler,
        int numberDevices,
        const size_t& nWorkerThreads,
        const size_t& nIOThreads,
        BackendConfig ioconfig,
        BackendConfig workerconfig
    ) const;

};
