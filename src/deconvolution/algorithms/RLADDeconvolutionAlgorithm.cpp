#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"
#include "UtlFFT.h"
#include "UtlImage.h"
#include <iostream>

void RLADDeconvolutionAlgorithm::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    // Configure algorithm-specific parameters
    iterations = config.iterations;
    dampingDecrease = 0; // Fixed to exponential decay as in original
    alpha = 0.9;         // Fixed as in original
    beta = 0.01;         // Fixed as in original
}

void RLADDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    BaseDeconvolutionAlgorithmCPU::configure(config);
    
    // Configure algorithm-specific parameters
    configureAlgorithmSpecific(config);
}

bool RLADDeconvolutionAlgorithm::preprocessBackendSpecific(int channel_num, int psf_index) {
    // No specific preprocessing needed for RLAD algorithm
    return true;
}

void RLADDeconvolutionAlgorithm::algorithmBackendSpecific(int channel_num, complex* H, complex* g, complex* f) {
    // Allocate memory for intermediate arrays using base class helper functions
    complex *c = nullptr;
    if (!allocateCPUArray(c, cubeVolume)) {
        std::cerr << "[ERROR] Failed to allocate memory for RLAD algorithm processing" << std::endl;
        return;
    }
    
    // Initialize result with input data
    copyComplexArray(g, f, cubeVolume);

    double a;

    for (int n = 0; n < iterations; ++n) {
        // Calculate damping factor
        if (dampingDecrease == 0) {
            a = alpha * exp(-beta * n);
        } else {  // Linear decay
            a = alpha - beta * n;
        }

        std::cout << "\r[STATUS] Channel: " << channel_num + 1 << " GridImage: " << totalGridNum 
                  << "/" << gridImages.size() << " Iteration: " << n + 1 << "/" << iterations << " ";

        // a) First transformation: Fn = FFT(fn)
        if (!executeForwardFFT(f, c)) {
            std::cerr << "[ERROR] Forward FFT failed in RLAD algorithm" << std::endl;
            goto cleanup;
        }

        // Fn' = Fn * H
        UtlFFT::complexMultiplication(f, H, c, cubeVolume);

        // fn' = IFFT(Fn')
        if (!executeBackwardFFT(c, f)) {
            std::cerr << "[ERROR] Backward FFT failed in RLAD algorithm" << std::endl;
            goto cleanup;
        }
        UtlFFT::octantFourierShift(f, cubeWidth, cubeHeight, cubeDepth);

        // b) Calculation of the Correction Factor: c = g / fn'
        UtlFFT::complexDivision(g, f, c, cubeVolume, epsilon);

        // c) Second transformation: C = FFT(c)
        if (!executeForwardFFT(c, f)) {
            std::cerr << "[ERROR] Forward FFT of correction factor failed" << std::endl;
            goto cleanup;
        }

        // C' = C * conj(H)
        UtlFFT::complexMultiplicationWithConjugate(f, H, f, cubeVolume);

        // c' = IFFT(C')
        if (!executeBackwardFFT(f, c)) {
            std::cerr << "[ERROR] Backward FFT of correction failed" << std::endl;
            goto cleanup;
        }
        UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);

        // d) Update the estimated image:
        // fn = IFFT(Fn)
        if (!executeBackwardFFT(c, f)) {
            std::cerr << "[ERROR] Final backward FFT failed in RLAD algorithm" << std::endl;
            goto cleanup;
        }

        // c = c * a (Apply adaptive damping)
        UtlFFT::scalarMultiplication(c, a, c, cubeVolume);

        // fn+1' = fn * c
        UtlFFT::complexMultiplication(f, c, f, cubeVolume);

        // Debugging check
        if (!validateComplexArray(f, cubeVolume, "RLAD result")) {
            std::cout << "[WARNING] Invalid array values detected in iteration " << n + 1 << std::endl;
        }

        std::flush(std::cout);
    }
    
    // Clean up allocated arrays
cleanup:
    deallocateCPUArray(c);
}

bool RLADDeconvolutionAlgorithm::postprocessBackendSpecific(int channel_num, int psf_index) {
    // No specific postprocessing needed for RLAD algorithm
    return true;
}

bool RLADDeconvolutionAlgorithm::allocateBackendMemory(int channel_num) {
    // No specific memory allocation needed for RLAD algorithm
    return true;
}

void RLADDeconvolutionAlgorithm::deallocateBackendMemory(int channel_num) {
    // No specific memory deallocation needed for RLAD algorithm
}

void RLADDeconvolutionAlgorithm::cleanupBackendSpecific() {
    // No specific cleanup needed for RLAD algorithm
}

// Legacy algorithm method for compatibility with existing code
void RLADDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, complex* H, complex* g, complex* f) {
    // Simply delegate to the new backend-specific implementation
    algorithmBackendSpecific(channel_num, H, g, f);
}
