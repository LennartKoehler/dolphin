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
#include <algorithm>
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
 * on the device). Reports min(ioThreads, numTasks) staging sets: with fewer
 * tasks than io threads, only one set can ever be in flight per task.
 */
class IOStagingMemoryComponent : public MemoryComponent {
public:
    explicit IOStagingMemoryComponent(size_t ioThreads) : ioThreads(ioThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(ioThreads, numTasks);
        const size_t volumeBytes = cubeShape.getVolume() * sizeof(real_t);
        return MemoryRequirement{instances * volumeBytes, instances * 2 * volumeBytes};
    }

    std::string name() const override { return "io staging"; }

private:
    size_t ioThreads;
};

/**
 * @brief MemoryComponent for the padding transient (no owning class exists).
 *
 * While a cube is padded straight into the io staging input buffer, the
 * unpadded source cube is held alongside it (the padded target IS the staging
 * buffer, accounted by the io staging component). Runs per io thread, so it
 * is capped at min(ioThreads, numTasks) like the staging buffers it
 * accompanies.
 */
class PaddingPeakMemoryComponent : public MemoryComponent {
public:
    PaddingPeakMemoryComponent(const Padding& cubePadding, size_t ioThreads)
        : cubePadding(cubePadding), ioThreads(ioThreads) {}

    MemoryRequirement footprint(const CuboidShape& paddedCube, size_t numTasks) const override {
        const size_t paddingVolume = cubePadding.getTotalPadding().getVolume();
        const size_t unpaddedVolume = paddedCube.getVolume() > paddingVolume
            ? paddedCube.getVolume() - paddingVolume
            : paddedCube.getVolume();
        const size_t bytes = unpaddedVolume * sizeof(real_t);
        return MemoryRequirement{std::min(ioThreads, numTasks) * bytes, 0};
    }

    std::string name() const override { return "padding"; }

private:
    Padding cubePadding;
    size_t ioThreads;
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
    /// run. The externally-owned components (reader, writer, psf cache) are
    /// borrowed; the strategy-owned terms (io staging one set per io thread,
    /// the algorithm's memory component, padding peak) are created and kept
    /// alive by the returned footprint. The backend's memory is NOT added
    /// here: the algorithm's memory component funnels the backend manager's
    /// FFT plan query for its own plan set. Each per-thread term caps its
    /// instances at the plan's task count via MemoryComponent::footprint.
    /// Used by createPlan and by tests to inspect the modeled demand.
    static MemoryFootprint buildMemoryFootprint(
        const MemoryComponent& readerComponent,
        const MemoryComponent& writerComponent,
        const IBackendManager& backendManager,
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

    /// Which global background strategy applies to this run, if any: the
    /// algorithm estimates its background, the config asks for a global level
    /// (not per-tile) and no explicit level is configured.
    static std::optional<BackgroundStrategyType> globalBackgroundStrategy(
        const DeconvolutionAlgorithm& algorithm,
        const DeconvolutionConfig& config);

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
