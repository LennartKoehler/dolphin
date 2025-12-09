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
    
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Richardson-Lucy algorithm" << std::endl;
        return;
    }
    int iterations = 1;
    // Allocate memory for intermediate arrays
    assert(backend->getMemoryManager().isOnDevice(f.data) && "PSF is not on device");
    ComplexData c = backend->getMemoryManager().allocateMemoryOnDevice(g.size);
    backend->getMemoryManager().memCopy(g, f);

    for (int n = 0; n < iterations; ++n) {

        // a) First transformation: Fn = FFT(fn)
        backend->getDeconvManager().forwardFFT(f, f);

        // Fn' = Fn * H
        backend->getDeconvManager().complexMultiplication(f, H, c);

        // fn' = IFFT(Fn') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);
        backend->getDeconvManager().scalarMultiplication(c, 1.0 / g.size.volume, c); // Add normalization


        // b) Calculation of the Correction Factor: c = g / fn'
        backend->getDeconvManager().complexDivision(g, c, c, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        backend->getDeconvManager().forwardFFT(c, c);

        // // C' = C * conj(H)
        backend->getDeconvManager().complexMultiplicationWithConjugate(c, H, c);

        // // c' = IFFT(C') + NORMALIZE
        backend->getDeconvManager().backwardFFT(c, c);
        backend->getDeconvManager().scalarMultiplication(c, 1.0 / g.size.volume, c); // Add normalization


        backend->getDeconvManager().backwardFFT(f, f);
        backend->getDeconvManager().scalarMultiplication(f, 1.0 / g.size.volume, f); // Add normalization

        backend->getDeconvManager().complexMultiplication(f, c, f);
 
    }
    // backend->getMemoryManager().freeMemoryOnDevice(c); // dont need because it is managed within complexdatas destructor
}
std::unique_ptr<DeconvolutionAlgorithm> TestAlgorithm::cloneSpecific() const{return std::make_unique<TestAlgorithm>();}

size_t TestAlgorithm::getMemoryMultiplier() const {
    return 3; // No additional memory allocation + 3 input copies
}
