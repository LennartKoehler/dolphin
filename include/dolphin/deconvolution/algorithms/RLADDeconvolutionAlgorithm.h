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
    size_t getWorkingSetBytes(const CuboidShape& dataSize) const override;
    // Same plan set as RichardsonLucy: out-of-place forward r2c, in-place
    // backward c2r, and the second-pass r2c in-place on the c buffer
    // (c_complex reinterprets c).
    std::vector<FFTPlanDescription> getFFTPlanDescriptions(const CuboidShape& dataSize) const override {
        return {
            {PlanDirection::FORWARD, PlanType::REAL, dataSize, false},
            {PlanDirection::BACKWARD, PlanType::REAL, dataSize, true},
            {PlanDirection::FORWARD, PlanType::REAL, dataSize, true},
        };
    }
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
