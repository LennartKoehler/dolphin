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
    size_t getWorkingSetBytes(const CuboidShape& dataSize) const override;

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