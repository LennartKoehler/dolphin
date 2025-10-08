#include "deconvolution/algorithms/TestAlgorithm.h"



void TestAlgorithm::configure(const DeconvolutionConfig& config){}
void TestAlgorithm::deconvolve(const ComplexData& H, const ComplexData& g, ComplexData& f){
    backend->memCopy(g, f);
    // backend->hasNAN(f);
}
std::unique_ptr<DeconvolutionAlgorithm> TestAlgorithm::clone() const{return std::make_unique<TestAlgorithm>();}

size_t TestAlgorithm::getMemoryMultiplier() const {
    return 3; // No additional memory allocation + 3 input copies
}
