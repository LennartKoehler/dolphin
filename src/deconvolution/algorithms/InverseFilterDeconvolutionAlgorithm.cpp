#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include "UtlFFT.h"
#include "UtlImage.h"
#include <iostream>

void InverseFilterDeconvolutionAlgorithm::configureAlgorithmSpecific(const DeconvolutionConfig& config) {
    // No specific configuration needed for this simple algorithm
}

void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    BaseDeconvolutionAlgorithmCPU::configure(config);
    
    // Configure algorithm-specific parameters
    configureAlgorithmSpecific(config);
}

bool InverseFilterDeconvolutionAlgorithm::preprocessBackendSpecific(int channel_num, int psf_index) {
    // No specific preprocessing needed for inverse filter
    return true;
}

void InverseFilterDeconvolutionAlgorithm::algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    // Forward FFT on image
    fftw_complex* temp_g = nullptr;
    if (!allocateCPUArray(temp_g, cubeVolume)) {
        std::cerr << "[ERROR] Failed to allocate memory for inverse filter processing" << std::endl;
        return;
    }
    
    try {
        // Copy input data to working array
        copyComplexArray(g, temp_g, cubeVolume);
        
        // Forward FFT on image
        if (!executeForwardFFT(temp_g, g)) {
            std::cerr << "[ERROR] Forward FFT failed in inverse filter algorithm" << std::endl;
            return;
        }

        // Division in frequency domain
        UtlFFT::complexDivisionStabilized(g, H, f, cubeVolume, epsilon);

        // Inverse FFT
        if (!executeBackwardFFT(f, temp_g)) {
            std::cerr << "[ERROR] Backward FFT failed in inverse filter algorithm" << std::endl;
            return;
        }
        UtlFFT::octantFourierShift(temp_g, cubeWidth, cubeHeight, cubeDepth);
        
        // Copy result to output
        copyComplexArray(temp_g, f, cubeVolume);
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in inverse filter algorithm: " << e.what() << std::endl;
    }
    
    // Cleanup temporary arrays
    deallocateCPUArray(temp_g);
}

bool InverseFilterDeconvolutionAlgorithm::postprocessBackendSpecific(int channel_num, int psf_index) {
    // No specific postprocessing needed for inverse filter
    return true;
}

bool InverseFilterDeconvolutionAlgorithm::allocateBackendMemory(int channel_num) {
    // No specific memory allocation needed for inverse filter
    return true;
}

void InverseFilterDeconvolutionAlgorithm::deallocateBackendMemory(int channel_num) {
    // No specific memory deallocation needed for inverse filter
}

void InverseFilterDeconvolutionAlgorithm::cleanupBackendSpecific() {
    // No specific cleanup needed for inverse filter
}

// Legacy algorithm method for compatibility with existing code
void InverseFilterDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    // Simply delegate to the new backend-specific implementation
    algorithmBackendSpecific(channel_num, H, g, f);
}
