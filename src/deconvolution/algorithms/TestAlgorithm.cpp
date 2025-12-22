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

#include "deconvolution/algorithms/TestAlgorithm.h"



void TestAlgorithm::configure(const DeconvolutionConfig& config) {
    // Test algorithm has no configuration parameters
}

void TestAlgorithm::init(const RectangleShape& dataSize) {
    // Test algorithm doesn't need any special initialization or memory allocation
    initialized = true;
}

bool TestAlgorithm::isInitialized() const {
    return initialized;
}

void TestAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    backend->getMemoryManager().memCopy(H, f); 
    
    // Test algorithm implementation - placeholder that does nothing
}

std::unique_ptr<DeconvolutionAlgorithm> TestAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<TestAlgorithm>();
    copy->initialized = false; // Clone needs to be re-initialized
    return copy;
}

size_t TestAlgorithm::getMemoryMultiplier() const {
    return 3; // No additional memory allocation + 3 input copies
}
