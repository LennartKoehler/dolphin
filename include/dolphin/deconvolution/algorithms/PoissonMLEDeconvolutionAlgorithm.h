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

// Poisson / I-divergence (photon-counting) likelihood deconvolution with the
// plain Richardson-Lucy unit step, classic in-core data step:
//   DATA STEP       m = Hf + bg, ratio R = H^T(g/m), f <- max(f*R, 0)
//                   (one shared work buffer carries m and R; they are never
//                   needed simultaneously in the unit-step path)
//   PRIOR + SIEVE   see PoissonMLECore
//
// Sibling variants: PoissonMLEStreamed (out-of-core data step, lower VRAM
// peak, extra PCIe traffic) and PoissonMLEAdaptiveStep (exact Newton line
// search). They share the math through PoissonMLECore, not inheritance.
class PoissonMLEDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    PoissonMLEDeconvolutionAlgorithm() = default;
    ~PoissonMLEDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const CuboidShape& dataSize) override;
    void preAllocate(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    std::unique_ptr<const MemoryComponent> getMemoryComponent(
        const IBackendManager& backendManager, size_t workerThreads) const override;

protected:
    PoissonMLECore core_;

private:
    // One data step: m = Hf + bg, R = H^T(g/m) (both in the shared work
    // buffer), then the fused update f = max(f*R, 0).
    void dataStep(const IComputeBackend& compute,
                  const ComplexData& H, const RealData& g, real_t bg, RealData& f);

    bool initialized = false;

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};



/**
 * @brief MemoryComponent of the classic PoissonMLE: one real in-place-FFT
 * scratch buffer (m -> ratio per iteration, plus the sieve's OTF buffer) per
 * worker thread, plus the FFT work-area memory of the variant's plan set.
 *
 * One working set per worker thread, capped at min(workerThreads, numTasks)
 * like every per-thread system, always reported on the DEVICE plane —
 * MemoryBudget::fits folds the device bytes into the host pool when the
 * backend is the CPU (device == host), so no placement flag is needed.
 * The FFT plan memory comes from ONE backend-manager query for the whole
 * plan set (the backend's sharing policy applies inside, e.g. CUDA's shared
 * work area counts the largest plan instead of the sum) and is folded into
 * the same per-thread concurrency — one backend instance per worker thread
 * holds the plans' work areas.
 */
class PoissonMLEAlgorithmMemoryComponent : public MemoryComponent {
public:
    PoissonMLEAlgorithmMemoryComponent(const IBackendManager& backendManager,
                                       size_t workerThreads)
        : backendManager(backendManager), workerThreads(workerThreads) {}

    MemoryRequirement footprint(const CuboidShape& cubeShape, size_t numTasks) const override {
        const size_t instances = std::min(workerThreads, numTasks);
        const size_t scratchBytes = cubeShape.getVolume() * sizeof(real_t);
        const size_t planBytes = backendManager.getFFTPlanMemoryBytes(
            PoissonMLECore::planDescriptions(cubeShape), workerThreads);
        return MemoryRequirement{0, instances * scratchBytes + planBytes};
    }

    std::string name() const override { return "algorithm working set + fft plans"; }

private:
    const IBackendManager& backendManager;
    size_t workerThreads;
};

