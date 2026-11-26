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
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

/**
 * @brief Values the plan resolves once for the whole run and every cube needs.
 *
 * Per-run (not per-subimage) quantities live here instead of being estimated
 * inside deconvolve() from the tile, where they would differ between
 * neighbouring cubes and leave visible seams. The plan (strategy) fills it;
 * algorithms that do not need a field ignore it.
 */
struct DeconvolutionRunContext {
    // Flat background of the whole input image (low percentile of the data).
    // std::nullopt: not available (per-tile estimate / explicit override).
    std::optional<float> background;
};

class DeconvolutionAlgorithm{
    using prFunction = std::function<void(int)>;
public:
    DeconvolutionAlgorithm() = default;
    virtual ~DeconvolutionAlgorithm() = default;

    /**
     * @brief Configure the algorithm from its own config class.
     *
     * The factory entry point (see DeconvolutionAlgorithmFactory::create) hands
     * the base reference to configure(const DeconvolutionConfig&), which
     * downcasts to the algorithm's own config class and forwards here. Each
     * algorithm therefore reads exactly the parameters its config registers.
     */
    virtual void configure(const DeconvolutionConfig& config) = 0;


    // Initialize algorithm-specific data allocations.
    //
    // Two-phase on purpose, so the caller can choose WHEN the two kinds of
    // backend state are created:
    //
    //  init()        creates the FFT plans the algorithm will use. On a device
    //                backend this transiently needs a plan work area, so calling
    //                init() before the io buffers are allocated keeps that
    //                transient from overlapping the image buffers (lower peak).
    //                Must be called AFTER setBackend().
    //  preAllocate() allocates the algorithm's persistent scratch arrays and any
    //                views that alias them. Call after init().
    //
    // Deconvolve() requires BOTH to have run. The default preAllocate() is a
    // no-op for algorithms whose init() allocates nothing.
    virtual void init(const CuboidShape& dataSize) = 0;
    virtual void preAllocate(const CuboidShape& dataSize) {}
    virtual bool isInitialized() const = 0;

    // it is assumed that the input of convolve is already located on the backend device
    // `runContext` carries values resolved once for the whole run (e.g. the
    // image-wide background) so that per-subimage calls stay consistent.
    virtual void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                            const DeconvolutionRunContext& runContext = {}) = 0;

    // Peak bytes this algorithm additionally holds for a cube of the given
    // (padded) shape: every scratch array allocated in init() or during
    // deconvolve(), beyond the input/output buffers passed in. Purely the
    // SIZE; where those bytes live (host or device) is decided by the
    // strategy, see AlgorithmMemoryComponent.
    virtual size_t getWorkingSetBytes(const CuboidShape& dataSize) const = 0;

    // The FFT plan descriptions this algorithm will use for a cube of
    // dataSize — mirroring the plans created in init() (kept in sync with
    // it, including lazily created ones). Queried by getMemoryComponent to
    // model the FFT work-area memory before any backend exists. Default: none.
    virtual std::vector<FFTPlanDescription> getFFTPlanDescriptions(const CuboidShape& dataSize) const {
        return {};
    }

    // The algorithm's memory demand for one run as a MemoryComponent: one
    // scratch working set (getWorkingSetBytes) plus the FFT work-area memory
    // of the plan set above, per worker thread. The plan memory is queried
    // from the backend manager for the whole set at once, so the backend's
    // own sharing policy applies inside (e.g. CUDA's shared work area counts
    // the largest plan, not the sum). The strategy adds THIS component to
    // the run's footprint; the backend's FFT memory is never added separately.
    std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, bool onDevice, size_t workerThreads) const;

    void setBackend(IBackend& backend){this->backend = &backend;}

    inline std::unique_ptr<DeconvolutionAlgorithm> clone() const{
        std::unique_ptr<DeconvolutionAlgorithm> clone = cloneSpecific();
        if (progressFunction)
            clone->setProgressTracker(progressFunction);
        return clone;
    }

    void setProgressTracker(prFunction progressFunction){ this->progressFunction = progressFunction; }
protected:
    virtual std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const = 0;

    /**
     * @brief Downcast a base config to the algorithm's config class.
     * @throws std::runtime_error naming both the algorithm and the config class
     */
    template <typename DerivedConfig>
    static const DerivedConfig& asConfig(const DeconvolutionConfig& config, const char* algorithmName) {
        const auto* typed = dynamic_cast<const DerivedConfig*>(&config);
        if (!typed) {
            throw std::runtime_error(std::string("Algorithm '") + algorithmName +
                                     "' requires a " + DerivedConfig::kAlgorithmName +
                                     " config, got: " + config.getName());
        }
        return *typed;
    }

    // Shared numerical guard for division-based updates (set from the config's
    // min_denominator by the algorithms that use it).
    float min_denominator = 1e-6f;

    prFunction progressFunction;
    IBackend* backend;


};

/**
 * @brief MemoryComponent of one deconvolution algorithm: its scratch working
 * sets plus the FFT work-area memory of the plans it will create.
 *
 * One working set per worker thread, capped at min(workerThreads, numTasks)
 * like every per-thread system, placed on the plane the strategy chooses
 * (the compute device; host RAM for the CPU backend where device == host).
 * The FFT plan memory comes from ONE backend-manager query for the whole
 * plan set (the backend's sharing policy applies inside, e.g. CUDA's shared
 * work area counts the largest plan instead of the sum) and is folded into
 * the same per-thread concurrency — one backend instance per worker thread
 * holds the plans' work areas.
 */
class AlgorithmMemoryComponent : public MemoryComponent {
public:
    AlgorithmMemoryComponent(const DeconvolutionAlgorithm& algorithm,
                             const IBackendManager& backendManager,
                             bool onDevice, size_t workerThreads)
        : algorithm(algorithm), backendManager(backendManager),
          onDevice(onDevice), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        size_t bytes = instances * algorithm.getWorkingSetBytes(cubeShape);
        bytes += instances * backendManager.getFFTPlanMemoryBytes(
            algorithm.getFFTPlanDescriptions(cubeShape));
        return onDevice ? MemoryRequirement{0, bytes} : MemoryRequirement{bytes, 0};
    }

    std::string name() const override { return "algorithm working set + fft plans"; }

private:
    const DeconvolutionAlgorithm& algorithm;
    const IBackendManager& backendManager;
    bool onDevice;
    size_t workerThreads;
};

inline std::unique_ptr<const MemoryComponent> DeconvolutionAlgorithm::getMemoryComponent(
    const IBackendManager& backendManager, bool onDevice, size_t workerThreads) const {
    return std::make_unique<AlgorithmMemoryComponent>(*this, backendManager, onDevice, workerThreads);
}
