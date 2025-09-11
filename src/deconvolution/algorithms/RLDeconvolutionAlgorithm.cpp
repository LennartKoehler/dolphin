#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "UtlFFT.h"
#include <iostream>
#include <omp.h>

void RLDeconvolutionAlgorithm::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    // Configure algorithm-specific parameters
    iterations = config.iterations;

    // Output algorithm-specific configuration
    std::cout << "[CONFIGURATION] Richardson-Lucy algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << iterations << std::endl;
}

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    BaseDeconvolutionAlgorithmCPU::configure(config);
    
    // Configure algorithm-specific parameters
    configureAlgorithmSpecific(config);
}

bool RLDeconvolutionAlgorithm::preprocessBackendSpecific(int channel_num, int psf_index) {
    // Richardson-Lucy specific preprocessing if needed
    return true;
}

void RLDeconvolutionAlgorithm::algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    // Allocate memory for intermediate FFTW arrays - using base class helper functions
    fftw_complex *c = nullptr;
    if (!allocateCPUArray(c, cubeVolume)) {
        std::cerr << "[ERROR] Failed to allocate memory for CPU Richardson-Lucy processing" << std::endl;
        return;
    }
    
    // Initialize result with input data
    copyComplexArray(g, f, cubeVolume);

    for (int n = 0; n < iterations; ++n) {
        std::cout << "\r[STATUS] Channel: " << channel_num + 1 << " GridImage: " << totalGridNum
                  << "/" << gridImages.size() << " Iteration: " << n + 1 << "/" << iterations << " ";

        // a) First transformation:Fn = FFT(fn)
        if (!executeForwardFFT(f, c)) {
            std::cerr << "[ERROR] Forward FFT failed in Richardson-Lucy algorithm" << std::endl;
            deallocateCPUArray(c);
            return;
        }
        
        // Fn' = Fn * H
        UtlFFT::complexMultiplication(f, H, c, cubeVolume);

        // fn' = IFFT(Fn')
        if (!executeBackwardFFT(c, f)) {
            std::cerr << "[ERROR] Backward FFT failed in Richardson-Lucy algorithm" << std::endl;
            deallocateCPUArray(c);
            return;
        }
        UtlFFT::octantFourierShift(f, cubeWidth, cubeHeight, cubeDepth);

        // b) Calculation of the Correction Factor: c = g / fn'
        UtlFFT::complexDivision(g, f, c, cubeVolume, epsilon);

        // c) Second transformation: C = FFT(c)
        if (!executeForwardFFT(c, f)) {
            std::cerr << "[ERROR] Forward FFT of correction factor failed" << std::endl;
            deallocateCPUArray(c);
            return;
        }

        // C' = C * conj(H)
        UtlFFT::complexMultiplicationWithConjugate(f, H, f, cubeVolume);

        // c' = IFFT(C')
        if (!executeBackwardFFT(f, c)) {
            std::cerr << "[ERROR] Backward FFT of correction failed" << std::endl;
            deallocateCPUArray(c);
            return;
        }
        UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);

        // d) Update the estimated image: fn+1 = fn * c
        if (!copyComplexArray(c, f, cubeVolume)) {
            std::cerr << "[ERROR] Failed to copy correction factor in Richardson-Lucy algorithm" << std::endl;
            deallocateCPUArray(c);
            return;
        }

        // Debugging check
        if (!validateComplexArray(f, cubeVolume, "Richardson-Lucy result")) {
            std::cout << "[WARNING] Invalid array values detected in iteration " << n + 1 << std::endl;
        }

        std::flush(std::cout);
    }
    
    // Cleanup temporary arrays
    deallocateCPUArray(c);
}

bool RLDeconvolutionAlgorithm::postprocessBackendSpecific(int channel_num, int psf_index) {
    // Richardson-Lucy specific postprocessing if needed
    return true;
}

bool RLDeconvolutionAlgorithm::allocateBackendMemory(int channel_num) {
    // Allocate memory specific to Richardson-Lucy algorithm if needed
    return true;
}

void RLDeconvolutionAlgorithm::deallocateBackendMemory(int channel_num) {
    // Deallocate memory specific to Richardson-Lucy algorithm if needed
}

void RLDeconvolutionAlgorithm::cleanupBackendSpecific() {
    // Cleanup specific to Richardson-Lucy algorithm if needed
}

// Legacy algorithm method for compatibility with existing code
void RLDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    // Simply delegate to the new backend-specific implementation
    algorithmBackendSpecific(channel_num, H, g, f);
}
