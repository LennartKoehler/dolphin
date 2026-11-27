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

#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/configs/InverseFilterConfig.h"
#include <memory>

class InverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    InverseFilterDeconvolutionAlgorithm() = default;
    ~InverseFilterDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void configure(const InverseFilterConfig& config);
    void init(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, size_t workerThreads) const override;
private:
    bool initialized = false;

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};

/**
 * @brief MemoryComponent of the InverseFilter: no scratch of its own (it
 * transforms in the caller's buffers), so the component carries only the FFT
 * work-area memory of its out-of-place forward/backward real plan pair — one
 * instance set per worker thread, on the plane the strategy chooses.
 */
class InverseFilterAlgorithmMemoryComponent : public MemoryComponent {
public:
    InverseFilterAlgorithmMemoryComponent(const IBackendManager& backendManager,
                                          size_t workerThreads)
        : backendManager(backendManager), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        const std::vector<FFTPlanDescription> plans = {
            {PlanDirection::FORWARD, PlanType::REAL, cubeShape, false},
            {PlanDirection::BACKWARD, PlanType::REAL, cubeShape, false},
        };
        return MemoryRequirement{0, backendManager.getFFTPlanMemoryBytes(plans, workerThreads)};
    }

    std::string name() const override { return "algorithm working set + fft plans"; }

private:
    const IBackendManager& backendManager;
    size_t workerThreads;
};
