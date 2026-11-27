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
#include "dolphin/deconvolution/configs/RichardsonLucyADConfig.h"
#include <memory>

class RLADDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    RLADDeconvolutionAlgorithm() = default;
    ~RLADDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void configure(const RichardsonLucyADConfig& config);
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
    int dampingDecrease; //0=exp, 1=lin
    double alpha;
    double beta;
    bool initialized = false;

    // Algorithm-specific data members for intermediate calculations
    ComplexData c_complex;  // Frequency domain
    RealData c;             // Spatial domain (real-valued)

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};

/**
 * @brief MemoryComponent of RichardsonLucy with adaptive damping: one real
 * volume per worker thread (the c buffer), plus the FFT work-area memory of
 * the same plan set RichardsonLucy uses. Same concurrency and plane rules as
 * RLAlgorithmMemoryComponent.
 */
class RLADAlgorithmMemoryComponent : public MemoryComponent {
public:
    RLADAlgorithmMemoryComponent(const IBackendManager& backendManager,
                                 size_t workerThreads)
        : backendManager(backendManager), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        const size_t scratchBytes = cubeShape.getVolume() * sizeof(real_t);
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
