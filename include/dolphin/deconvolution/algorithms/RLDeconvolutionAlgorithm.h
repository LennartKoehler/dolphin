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
#include "dolphin/deconvolution/configs/RichardsonLucyConfig.h"
#include <memory>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    RLDeconvolutionAlgorithm() = default;
    ~RLDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void configure(const RichardsonLucyConfig& config);
    void init(const CuboidShape& dataSize) override;
    void preAllocate(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, size_t workerThreads) const override;
private:
    // Configuration (set in configure()); cloned in cloneSpecific().
    int iterations = 10;
    bool initialized = false;

    // Algorithm-specific data members for intermediate calculations
    ComplexData c_complex;
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};

/**
 * @brief MemoryComponent of RichardsonLucy: one complex working buffer
 * (2 real_t per element) per worker thread, plus the FFT work-area memory of
 * the plan set (out-of-place forward r2c, in-place backward c2r, and the
 * second-pass r2c which runs in-place on the c buffer). One working set per
 * worker thread, capped at min(workerThreads, numTasks), placed on the plane
 * the strategy chooses; the backend's sharing policy applies inside the plan
 * query (CUDA: one shared work area per backend instance, largest plan wins).
 */
class RLAlgorithmMemoryComponent : public MemoryComponent {
public:
    RLAlgorithmMemoryComponent(const IBackendManager& backendManager,
                               size_t workerThreads)
        : backendManager(backendManager), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        const size_t scratchBytes = 2 * cubeShape.getVolume() * sizeof(real_t);
        const std::vector<FFTPlanDescription> plans = {
            {PlanDirection::FORWARD, PlanType::REAL, cubeShape, false},
            {PlanDirection::BACKWARD, PlanType::REAL, cubeShape, true},
            {PlanDirection::FORWARD, PlanType::REAL, cubeShape, true},
        };
        return MemoryRequirement{0,
            instances * scratchBytes + backendManager.getFFTPlanMemoryBytes(plans, workerThreads)};
    }

    std::string name() const override { return "algorithm working set + fft plans"; }

private:
    const IBackendManager& backendManager;
    size_t workerThreads;
};
