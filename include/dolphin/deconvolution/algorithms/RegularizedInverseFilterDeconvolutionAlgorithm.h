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
#include "dolphin/deconvolution/configs/RegularizedInverseFilterConfig.h"
#include <memory>

class RegularizedInverseFilterDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    RegularizedInverseFilterDeconvolutionAlgorithm() = default;
    ~RegularizedInverseFilterDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void configure(const RegularizedInverseFilterConfig& config);
    void init(const CuboidShape& dataSize) override;
    void preAllocate(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, size_t workerThreads) const override;
private:
    // Configuration (set in configure()); cloned in cloneSpecific().
    double signal_to_noise = 10.0;
    bool initialized = false;

    // Algorithm-specific data members for intermediate calculations (frequency domain)
    ComplexData H2;
    ComplexData L;
    ComplexData L2;
    ComplexData FA;
    ComplexData FP;
    ComplexData f_complex;  // FFT of input image

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};

/**
 * @brief MemoryComponent of the RegularizedInverseFilter: FIVE real volumes
 * per worker thread (H2, L, L2, FA, FP, f_complex — FP reuses f_complex),
 * plus the FFT work-area memory of its out-of-place forward/backward real
 * plan pair. Same concurrency and plane rules as RLAlgorithmMemoryComponent.
 */
class RegularizedInverseFilterAlgorithmMemoryComponent : public MemoryComponent {
public:
    RegularizedInverseFilterAlgorithmMemoryComponent(const IBackendManager& backendManager,
                                                     size_t workerThreads)
        : backendManager(backendManager), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        const size_t scratchBytes = 5 * cubeShape.getVolume() * sizeof(real_t);
        const std::vector<FFTPlanDescription> plans = {
            {PlanDirection::FORWARD, PlanType::REAL, cubeShape, false},
            {PlanDirection::BACKWARD, PlanType::REAL, cubeShape, false},
        };
        return MemoryRequirement{0,
            instances * scratchBytes + backendManager.getFFTPlanMemoryBytes(plans, workerThreads)};
    }

    std::string name() const override { return "algorithm working set + fft plans"; }

private:
    const IBackendManager& backendManager;
    size_t workerThreads;
};
