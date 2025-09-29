#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"



class TestAlgorithm : public DeconvolutionAlgorithm{
void configure(const DeconvolutionConfig& config) override {}
void deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f) override {
    backend->memCopy(g, f);
}
std::unique_ptr<DeconvolutionAlgorithm> clone() const override {return std::make_unique<TestAlgorithm>();}
};