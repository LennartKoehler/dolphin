#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <memory>
#include <iostream>

class TestAlgorithm : public DeconvolutionAlgorithm {
public:
    TestAlgorithm() = default;
    virtual ~TestAlgorithm() = default;

    void configure(const DeconvolutionConfig& config) override;
    void init(const RectangleShape& dataSize) override;
    bool isInitialized() const override;
    void deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) override;
    size_t getMemoryMultiplier() const override;

private:
    std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override;
    bool initialized = false;
};

