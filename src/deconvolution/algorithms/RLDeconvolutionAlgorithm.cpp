#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include <iostream>
#include <omp.h>

RLDeconvolutionAlgorithm::RLDeconvolutionAlgorithm(std::shared_ptr<IDeconvolutionBackend> backend)
    : backend(backend) {
    std::cout << "[INFO] RLDeconvolutionAlgorithm initialized with backend" << std::endl;
}

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Call base class configure to set up common parameters
    iterations = config.iterations;


}

// Legacy algorithm method for compatibility with existing code
void RLDeconvolutionAlgorithm::deconvolve(fftw_complex* H, fftw_complex* g, fftw_complex* f, const RectangleShape& cubeSize) {
    if (!backend) {
        std::cerr << "[ERROR] No backend available for Richardson-Lucy algorithm" << std::endl;
        return;
    }
    int cubeVolume = cubeSize.volume;
    int cubeHeight = cubeSize.height;
    int cubeWidth = cubeSize.width;
    int cubeDepth = cubeSize.depth;

    // Allocate memory for intermediate arrays
    fftw_complex *c = nullptr;
    if (!allocateCPUArray(c, cubeVolume)) {
        std::cerr << "[ERROR] Failed to allocate memory for Richardson-Lucy processing" << std::endl;
        return;
    }
    
    // Initialize result with input data
    copyComplexArray(g, f, cubeVolume);

    for (int n = 0; n < iterations; ++n) {
        std::cout << "\r[STATUS] Iteration: " << iterations << " ";

        // a) First transformation:Fn = FFT(fn)
        backend->forwardFFT(f, c, cubeDepth, cubeHeight, cubeWidth);

        
        // Fn' = Fn * H
        backend->complexMultiplication(f, H, c, cubeVolume);

        // fn' = IFFT(Fn')
        backend->backwardFFT(c, f, cubeDepth, cubeHeight, cubeWidth);

        backend->octantFourierShift(f, cubeWidth, cubeHeight, cubeDepth);

        // b) Calculation of the Correction Factor: c = g / fn'
        backend->complexDivision(g, f, c, cubeVolume, complexDivisionEpsilon);

        // c) Second transformation: C = FFT(c)
        backend->forwardFFT(c, f, cubeDepth, cubeHeight, cubeWidth);


        // C' = C * conj(H)
        backend->complexMultiplicationWithConjugate(f, H, f, cubeVolume);

        // c' = IFFT(C')
        backend->backwardFFT(f, c, cubeDepth, cubeHeight, cubeWidth);

        backend->octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);

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



bool RLDeconvolutionAlgorithm::copyComplexArray(const fftw_complex* source, fftw_complex* destination, int size) {
    if (!source || !destination) {
        std::cerr << "[ERROR] Invalid array pointers in copyComplexArray" << std::endl;
        return false;
    }
    
    for (int i = 0; i < size; ++i) {
        destination[i][0] = source[i][0];  // Real part
        destination[i][1] = source[i][1];  // Imaginary part
    }
    return true;
}

bool RLDeconvolutionAlgorithm::validateComplexArray(fftw_complex* array, int size, const std::string& context) {
    if (!array) {
        std::cerr << "[ERROR] Invalid array pointer in validateComplexArray: " << context << std::endl;
        return false;
    }
    
    for (int i = 0; i < size; ++i) {
        // Check for NaN or infinite values
        if (!std::isfinite(array[i][0]) || !std::isfinite(array[i][1])) {
            std::cerr << "[ERROR] Invalid complex value detected in " << context << " at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

bool RLDeconvolutionAlgorithm::allocateCPUArray(fftw_complex*& array, int size) {
    if (size <= 0) {
        std::cerr << "[ERROR] Invalid size for array allocation" << std::endl;
        return false;
    }
    
    try {
        array = new fftw_complex[size];
        if (!array) {
            std::cerr << "[ERROR] Memory allocation failed" << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Memory allocation exception: " << e.what() << std::endl;
        return false;
    }
}

void RLDeconvolutionAlgorithm::deallocateCPUArray(fftw_complex* array) {
    if (array) {
        delete[] array;
        array = nullptr;
    }
}
