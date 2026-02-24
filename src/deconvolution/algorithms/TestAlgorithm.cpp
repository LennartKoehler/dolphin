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

#include "dolphin/deconvolution/algorithms/TestAlgorithm.h"
#include <iostream>
#include <cassert>
#include <spdlog/spdlog.h>

void TestAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    // No specific parameters for test algorithm
}

void TestAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for Test algorithm initialization");\
    
    initialized = true;
}

bool TestAlgorithm::isInitialized() const {
    return initialized;
}

void TestAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    assert(backend && "No backend available for Test algorithm");\
    
    assert(initialized && "Test algorithm not initialized. Call init() first.");\
    backend->getDeconvManager().complexMultiplicationWithConjugate(g, H, f);

    // Simple test: just copy H to f
    // backend->getDeconvManager().complexMultiplication()
    // backend->getMemoryManager().memCopy(H, f);
}

std::unique_ptr<DeconvolutionAlgorithm> TestAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<TestAlgorithm>();
    // Copy all relevant state
    copy->initialized = false; // Clone needs to be re-initialized
    // Don't copy backend - each thread needs its own
    return copy;
}

size_t TestAlgorithm::getMemoryMultiplier() const {
    return 0; // No additional memory allocation needed
}