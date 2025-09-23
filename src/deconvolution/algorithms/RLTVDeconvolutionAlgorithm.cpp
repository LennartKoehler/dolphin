#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include "UtlFFT.h"
#include <iostream>

void RLTVDeconvolutionAlgorithm::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    // Configure algorithm-specific parameters
    iterations = config.iterations;
    lambda = config.lambda;

    // Output algorithm-specific configuration
    std::cout << "[CONFIGURATION] Richardson-Lucy Total Variation algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << iterations << std::endl;
    std::cout << "[CONFIGURATION] lambda: " << lambda << std::endl;
}

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    BaseDeconvolutionAlgorithmCPU::configure(config);
    
    // Configure algorithm-specific parameters
    configureAlgorithmSpecific(config);
}

bool RLTVDeconvolutionAlgorithm::preprocessBackendSpecific(int channel_num, int psf_index) {
    // Richardson-Lucy TV specific preprocessing if needed
    return true;
}

void RLTVDeconvolutionAlgorithm::algorithmBackendSpecific(int channel_num, complex* H, complex* g, complex* f) {
    // Allocate memory for intermediate arrays using base class helper functions
    complex *c = nullptr;
    complex *gx = nullptr;
    complex *gy = nullptr;
    complex *gz = nullptr;
    complex *tv = nullptr;
    
    if (!allocateCPUArray(c, cubeVolume) ||
        !allocateCPUArray(gx, cubeVolume) ||
        !allocateCPUArray(gy, cubeVolume) ||
        !allocateCPUArray(gz, cubeVolume) ||
        !allocateCPUArray(tv, cubeVolume)) {
        std::cerr << "[ERROR] Failed to allocate memory for Richardson-Lucy TV algorithm" << std::endl;
        
        // Cleanup what we allocated
        deallocateCPUArray(c);
        deallocateCPUArray(gx);
        deallocateCPUArray(gy);
        deallocateCPUArray(gz);
        deallocateCPUArray(tv);
        return;
    }
    
    // Initialize result with input data
    copyComplexArray(g, f, cubeVolume);

    // Calculate gradients and the Total Variation (one-time computation)
    UtlFFT::gradientX(g, gx, cubeWidth, cubeHeight, cubeDepth);
    UtlFFT::gradientY(g, gy, cubeWidth, cubeHeight, cubeDepth);
    UtlFFT::gradientZ(g, gz, cubeWidth, cubeHeight, cubeDepth);
    UtlFFT::normalizeTV(gx, gy, gz, cubeWidth, cubeHeight, cubeDepth, epsilon);
    UtlFFT::gradientX(gx, gx, cubeWidth, cubeHeight, cubeDepth);
    UtlFFT::gradientY(gy, gy, cubeWidth, cubeHeight, cubeDepth);
    UtlFFT::gradientZ(gz, gz, cubeWidth, cubeHeight, cubeDepth);
    UtlFFT::computeTV(lambda, gx, gy, gz, tv, cubeWidth, cubeHeight, cubeDepth);

    for (int n = 0; n < iterations; ++n) {
        std::cout << "\r[STATUS] Channel: " << channel_num + 1 << " GridImage: " << totalGridNum
                  << "/" << gridImages.size() << " Iteration: " << n + 1 << "/" << iterations << " ";

        // a) First transformation: Fn = FFT(fn)
        if (!executeForwardFFT(f, c)) {
            std::cerr << "[ERROR] Forward FFT failed in Richardson-Lucy TV algorithm" << std::endl;
            goto cleanup;
        }

        // Fn' = Fn * H
        UtlFFT::complexMultiplication(f, H, c, cubeVolume);

        // fn' = IFFT(Fn')
        if (!executeBackwardFFT(c, f)) {
            std::cerr << "[ERROR] Backward FFT failed in Richardson-Lucy TV algorithm" << std::endl;
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

        // d) Update the estimated image:fn+1' = fn * c
        if (!copyComplexArray(c, f, cubeVolume)) {
            std::cerr << "[ERROR] Failed to update result image" << std::endl;
            goto cleanup;
        }

        // fn+1 = fn+1' * tv
        UtlFFT::complexMultiplication(f, tv, f, cubeVolume);

        // Debugging check
        if (!validateComplexArray(f, cubeVolume, "Richardson-Lucy TV result")) {
            std::cout << "[WARNING] Invalid array values detected in iteration " << n + 1 << std::endl;
        }

        std::flush(std::cout);
    }
    
    // Clean up allocated arrays
cleanup:
    deallocateCPUArray(c);
    deallocateCPUArray(gx);
    deallocateCPUArray(gy);
    deallocateCPUArray(gz);
    deallocateCPUArray(tv);
}

bool RLTVDeconvolutionAlgorithm::postprocessBackendSpecific(int channel_num, int psf_index) {
    // Richardson-Lucy TV specific postprocessing if needed
    return true;
}

bool RLTVDeconvolutionAlgorithm::allocateBackendMemory(int channel_num) {
    // Allocate memory specific to Richardson-Lucy TV algorithm if needed
    return true;
}

void RLTVDeconvolutionAlgorithm::deallocateBackendMemory(int channel_num) {
    // Deallocate memory specific to Richardson-Lucy TV algorithm if needed
}

void RLTVDeconvolutionAlgorithm::cleanupBackendSpecific() {
    // Cleanup specific to Richardson-Lucy TV algorithm if needed
}

// Legacy algorithm method for compatibility with existing code
void RLTVDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, complex* H, complex* g, complex* f) {
    // Simply delegate to the new backend-specific implementation
    algorithmBackendSpecific(channel_num, H, g, f);
}
