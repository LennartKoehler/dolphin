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

// PoissonMLE (RL unit step) with an out-of-core data step for VRAM-bound runs.
//
// Identical math and parameters to PoissonMLE; only WHERE the buffers live
// differs. The observed image g is captured on the HOST once per cube, the
// caller's device g buffer is reused as the data-step work buffer, and g is
// streamed back one z-slice per FFT stage. That removes the algorithm's own
// full-size work volume from the device: the peak is iterate + work + PSF
// (+ the shared cuFFT work area) instead of 4 volumes.
//
// Cost: one full-volume D2H download per cube plus one full-volume H2D upload
// PER ITERATION (slab-wise). On PCIe that is roughly one transform's worth of
// transfer time per iteration, so prefer the plain PoissonMLE whenever the
// cube fits in VRAM. On the CPU backend the streaming is free (host == device
// memory) and this variant behaves identically to PoissonMLE.
//
// Sibling of PoissonMLE / PoissonMLEAdaptiveStep; shares the math through
// PoissonMLECore, not inheritance.
class PoissonMLEStreamedDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    PoissonMLEStreamedDeconvolutionAlgorithm() = default;
    ~PoissonMLEStreamedDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const CuboidShape& dataSize) override;
    void preAllocate(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f,
                    const DeconvolutionRunContext& runContext = {}) override;
    size_t getWorkingSetBytes(const CuboidShape& dataSize) const override;
    std::vector<FFTPlanDescription> getFFTPlanDescriptions(const CuboidShape& dataSize) const override {
        // The streamed data step runs on the backend with the same in-place
        // plan pair the core declares (plus the out-of-place forward plan the
        // regular stages use), so the core's set is complete.
        return core_.planDescriptions(dataSize);
    }

private:
    // One z-slice of the padded in-place layout: staging buffer for the
    // slab-wise upload of g. The full work volume is the caller's g buffer.
    RealData ratioSlab;

    // Shared math + config of the PoissonMLE family (composition, not
    // inheritance). No classic work buffer is allocated here.
    PoissonMLECore core_;

    bool initialized = false;

    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};