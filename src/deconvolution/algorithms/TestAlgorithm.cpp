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



void TestAlgorithm::configure(const DeconvolutionConfig& config){}
void TestAlgorithm::deconvolve(const ComplexData& H, ComplexData& g, ComplexData& f) {
    

    // backend->getMemoryManager().freeMemoryOnDevice(c); // dont need because it is managed within complexdatas destructor
}
std::unique_ptr<DeconvolutionAlgorithm> TestAlgorithm::cloneSpecific() const{return std::make_unique<TestAlgorithm>();}

size_t TestAlgorithm::getMemoryMultiplier() const {
    return 3; // No additional memory allocation + 3 input copies
}
