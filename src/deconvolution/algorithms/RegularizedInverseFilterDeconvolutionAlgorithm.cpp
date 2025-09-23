#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "UtlFFT.h"
#include "UtlImage.h"
#include <iostream>

void RegularizedInverseFilterDeconvolutionAlgorithm::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    // Configure algorithm-specific parameters
    lambda = config.lambda;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    BaseDeconvolutionAlgorithmCPU::configure(config);
    
    // Configure algorithm-specific parameters
    configureAlgorithmSpecific(config);
}

bool RegularizedInverseFilterDeconvolutionAlgorithm::preprocessBackendSpecific(int channel_num, int psf_index) {
    // No specific preprocessing needed for regularized inverse filter
    return true;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::algorithmBackendSpecific(int channel_num, complex* H, complex* g, complex* f) {
    // Allocate memory for intermediate arrays using base class helper functions
    complex* H2 = nullptr;
    complex* L = nullptr;
    complex* L2 = nullptr;
    complex* FA = nullptr;
    complex* FP = nullptr;
    
    if (!allocateCPUArray(H2, cubeVolume) ||
        !allocateCPUArray(L, cubeVolume) ||
        !allocateCPUArray(L2, cubeVolume) ||
        !allocateCPUArray(FA, cubeVolume) ||
        !allocateCPUArray(FP, cubeVolume)) {
        std::cerr << "[ERROR] Failed to allocate memory for regularized inverse filter processing" << std::endl;
        
        // Cleanup what we allocated
        deallocateCPUArray(H2);
        deallocateCPUArray(L);
        deallocateCPUArray(L2);
        deallocateCPUArray(FA);
        deallocateCPUArray(FP);
        return;
    }
    
    try {
        // Forward FFT on image
        if (!executeForwardFFT(g, g)) {
            std::cerr << "[ERROR] Forward FFT failed in regularized inverse filter algorithm" << std::endl;
            return;
        }

        // H*H
        UtlFFT::complexMultiplication(H, H, H2, cubeVolume);
        
        // Laplacian L
        UtlFFT::calculateLaplacianOfPSF(H, L, cubeWidth, cubeHeight, cubeDepth);
        UtlFFT::complexMultiplication(L, L, L2, cubeVolume);
        UtlFFT::scalarMultiplication(L2, lambda, L2, cubeVolume);

        UtlFFT::complexAddition(H2, L2, FA, cubeVolume);
        UtlFFT::complexDivisionStabilized(H, FA, FP, cubeVolume, epsilon);
        UtlFFT::complexMultiplication(g, FP, f, cubeVolume);

        // Inverse FFT
        if (!executeBackwardFFT(f, g)) {
            std::cerr << "[ERROR] Backward FFT failed in regularized inverse filter algorithm" << std::endl;
            return;
        }
        UtlFFT::octantFourierShift(g, cubeWidth, cubeHeight, cubeDepth);
        
        // Copy result to output
        copyComplexArray(g, f, cubeVolume);
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in regularized inverse filter algorithm: " << e.what() << std::endl;
    }
    
    // Cleanup allocated arrays
    deallocateCPUArray(H2);
    deallocateCPUArray(L);
    deallocateCPUArray(L2);
    deallocateCPUArray(FA);
    deallocateCPUArray(FP);
}

bool RegularizedInverseFilterDeconvolutionAlgorithm::postprocessBackendSpecific(int channel_num, int psf_index) {
    // No specific postprocessing needed for regularized inverse filter
    return true;
}

bool RegularizedInverseFilterDeconvolutionAlgorithm::allocateBackendMemory(int channel_num) {
    // No specific memory allocation needed for regularized inverse filter
    return true;
}

void RegularizedInverseFilterDeconvolutionAlgorithm::deallocateBackendMemory(int channel_num) {
    // No specific memory deallocation needed for regularized inverse filter
}

void RegularizedInverseFilterDeconvolutionAlgorithm::cleanupBackendSpecific() {
    // No specific cleanup needed for regularized inverse filter
}

// Legacy algorithm method for compatibility with existing code
void RegularizedInverseFilterDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, complex* H, complex* g, complex* f) {
    // Simply delegate to the new backend-specific implementation
    algorithmBackendSpecific(channel_num, H, g, f);
}
