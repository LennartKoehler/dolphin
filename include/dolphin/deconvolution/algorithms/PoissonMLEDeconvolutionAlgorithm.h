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
#include <memory>

// Poisson / I-divergence (photon-counting) likelihood deconvolution: an alternating
// data/prior scheme on the objective
//     J(f) = sum [ g ln(g/m) - g + m ],   m = Hf + bg,   f >= bg.
// Each iteration:
//   DATA STEP  - gradient H^T(1 - g/m), direction d = -f*(1 - ratio), exact Newton
//                line search along f + alpha*d, additive update clipped at bg.
//   GUARD      - if J increased, damp the update (power) and redo the iteration.
//   PRIOR STEP - lambda-shrinkage f <- (sqrt(2*lambda*f + 1) - 1)/lambda + flux renormalize.
//   SIEVE      - annealed anisotropic Gaussian smoothing (roughness regularizer).
// Stopping is by iteration count (no quality-factor criterion).
class PoissonMLEDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    PoissonMLEDeconvolutionAlgorithm() = default;
    ~PoissonMLEDeconvolutionAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const CuboidShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, RealData& g, RealData& f) override;
    size_t getMemoryMultiplier() const override;

private:
    // Scratch (device) buffers. Each spatial quantity (model m, direction d, hd=H*d)
    // is a real in-place-FFT buffer; its ComplexView (same memory, non-owning) is the
    // FFT work buffer. FFTs follow RichardsonLucy: out-of-place real->complex, then
    // in-place complex->real into the same buffer's real view, so a real result is
    // never clobbered by an FFT.
    RealData model;       // forward model m = Hf + bg (spatial)
    RealData direction;   // search direction d = -f*(1 - H^T(g/m)) (spatial)
    RealData hd;          // H*d, the forward model of the direction (spatial)
    ComplexView modelC;   // complex view of `model` (FFT work buffer)
    ComplexView directionC; // complex view of `direction`
    ComplexView hdC;      // complex view of `hd`
    ComplexData hNorm;    // PSF OTF scaled to unit sum (DC = 1)
    ComplexData sieveOTF; // annealed anisotropic Gaussian low-pass kernel

    int iterations;
    float snr;
    float lambdaOverride;
    float backgroundOverride;
    bool useSieve;
    bool usePrior;
    int sievePasses;

    bool initialized = false;
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
};
