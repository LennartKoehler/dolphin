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
#include "dolphin/deconvolution/algorithms/PoissonMLECore.h"
#include <memory>

// PoissonMLE with an exact Newton line-search data step (guarded against the
// RL unit step) at the cost of one extra FFT (hd = H*d) plus ~30 J(alpha)
// evaluations per iteration, and per-iteration damping state:
//   DATA STEP   m = Hf + bg (kept alive for J(alpha));
//               d = f*(R - 1) with R = H^T(g/m)  (the search direction);
//               hd = H*d; line search along m(alpha) = m + alpha*hd;
//               f <- f + alpha*pwr*d, clipped at zero.
//   GUARD       if J increased, halve the damping power pwr and redo... (pwr
//               decays; reset to 1 per deconvolution).
//   PRIOR+SIEVE see PoissonMLECore (identical to the other variants).
//
// Sibling of PoissonMLE / PoissonMLEStreamed; shares the math through
// PoissonMLECore, not inheritance.
class PoissonMLEAdaptiveStepDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    PoissonMLEAdaptiveStepDeconvolutionAlgorithm() = default;
    ~PoissonMLEAdaptiveStepDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const CuboidShape& dataSize) override;
    void preAllocate(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, size_t workerThreads) const override;

private:
    // One line-search data step. The model m (in its own buffer) must stay
    // alive for J(alpha); the shared work buffer carries the direction d.
    void dataStep(const IComputeBackend& compute,
                  const ComplexData& H, const RealData& g, real_t bg, RealData& f);

    PoissonMLECore core_;

    // Extra line-search scratch. The unit-step variant shares one buffer for
    // m and d (never needed together there); this variant needs m (for
    // J(alpha)) and d simultaneously, so it keeps the model in its own buffer
    // and reuses core_.scratch for d. hd = H*d, the forward model of the
    // direction.
    RealData model;
    RealData hd;
    ComplexView hdC;

    // Per-deconvolution step-damping state.
    real_t pwr = 1.0f;

    bool initialized = false;

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};

/**
 * @brief MemoryComponent of the adaptive-step PoissonMLE: THREE real
 * in-place-FFT buffers per worker thread (the shared work buffer carrying the
 * direction d, the model m kept alive for J(alpha), and hd = H*d for the line
 * search), plus the FFT work-area memory of the variant's plan set. Same
 * concurrency and plane rules as PoissonMLEAlgorithmMemoryComponent.
 */
class PoissonMLEAdaptiveStepAlgorithmMemoryComponent : public MemoryComponent {
public:
    PoissonMLEAdaptiveStepAlgorithmMemoryComponent(const IBackendManager& backendManager,
                                                   size_t workerThreads)
        : backendManager(backendManager), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        const size_t scratchBytes = 3 * cubeShape.getVolume() * sizeof(real_t);
        const size_t planBytes = backendManager.getFFTPlanMemoryBytes(
            PoissonMLECore::planDescriptions(cubeShape), workerThreads);
        return MemoryRequirement{0, instances * scratchBytes + planBytes};
    }

    std::string name() const override { return "algorithm working set + fft plans"; }

private:
    const IBackendManager& backendManager;
    size_t workerThreads;
};
