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
#include "dolphinbackend/MemoryComponent.h"
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


    // The algorithm's memory demand for one run as a MemoryComponent. Each
    // algorithm composes its own component individually: it knows the scratch
    // arrays it allocates and the FFT plans it will create, and folds in the
    // FFT work-area memory by querying the backend manager for its plan set
    // (IBackendManager::getFFTPlanMemoryBytes — the backend's sharing policy
    // applies inside, e.g. CUDA's shared work area counts the largest plan,
    // not the sum). The component reports one instance set per worker thread
    // (capped at numTasks), always on the DEVICE plane — MemoryBudget::fits
    // folds the device bytes into the host pool when the backend is the CPU
    // (device == host), so no placement flag is needed. The strategy adds
    // THIS component to the run's footprint; the backend's FFT memory is
    // never added separately.
    virtual std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, size_t workerThreads) const = 0;

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

