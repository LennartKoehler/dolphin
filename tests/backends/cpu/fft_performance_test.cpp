#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphin/Logging.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>


void fillComplexDataWithRandom(ComplexData& result) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dist(-1.0, 1.0);
    std::uniform_real_distribution<> imag_dist(-1.0, 1.0);


    // Fill with random data
    for (int i = 0; i < result.getSize().getVolume(); ++i) {
        result.getData()[i][0] = real_dist(gen);  // Real part
        result.getData()[i][1] = imag_dist(gen);  // Imaginary part
    }

}

std::chrono::microseconds runLargeFFTTest(
    const std::string& backendName,
    IBackend& backend,
    IBackendMemoryManager& cpuMemManager,
    int totalElements,
    int num_iterations
) {
    std::cout << "\n--- Test 1: One large FFT ---" << std::endl;

    IDeconvolutionBackend& deconvBackend = backend.mutableDeconvManager();
    IBackendMemoryManager& memManager = backend.mutableMemoryManager();

    // Calculate dimensions for large FFT
    int largeDim = std::round(std::cbrt(totalElements));
    CuboidShape largeShape(largeDim, largeDim, largeDim);
    std::cout << "large FFT shape: " << largeShape.width << "x" << largeShape.height << "x" << largeShape.depth << std::endl;

    // Create and fill large data on CPU, then copy to test backend
    ComplexData largeCpuData = cpuMemManager.allocateMemoryOnDeviceComplex(largeShape);
    fillComplexDataWithRandom(largeCpuData);
    auto largeInput = memManager.copyDataToDevice(largeCpuData);
    auto largeOutput = memManager.allocateMemoryOnDeviceComplex(largeShape);

    // Free CPU data after copying to device
    cpuMemManager.freeMemoryOnDevice(largeCpuData);

    // Time the large FFT
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++){
        deconvBackend.forwardFFT(largeInput, largeOutput);
    }
    backend.sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto largeDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "large FFT time: " << largeDuration.count() << " mus" << std::endl;

    // Cleanup large data
    memManager.freeMemoryOnDevice(largeInput);
    memManager.freeMemoryOnDevice(largeOutput);

    return largeDuration;
}

std::chrono::microseconds runSmallFFTTest(
    const std::string& backendName,
    IBackend& backend,
    IBackendMemoryManager& cpuMemManager,
    int smallDim,
    int numSmallFFTs,
    int num_iterations
) {
    std::cout << "\n--- Test 2: " << numSmallFFTs << " Smaller FFTs ---" << std::endl;

    IDeconvolutionBackend& deconvBackend = backend.mutableDeconvManager();
    IBackendMemoryManager& memManager = backend.mutableMemoryManager();

    CuboidShape smallShape(smallDim, smallDim, smallDim);
    std::cout << "Small FFT shape: " << smallShape.width << "x" << smallShape.height << "x" << smallShape.depth << std::endl;

    // Create and fill small data on CPU, then copy to test backend
    ComplexData smallCpuData = cpuMemManager.allocateMemoryOnDeviceComplex(smallShape);
    fillComplexDataWithRandom(smallCpuData);
    std::vector<ComplexData> smallInput;
    std::vector<ComplexData> smallOutput;

    // Time the small FFTs
    auto start = std::chrono::high_resolution_clock::now();
    // Reserve space to avoid reallocations
    smallInput.reserve(numSmallFFTs);
    smallOutput.reserve(numSmallFFTs);

    // Create input and output data for each small FFT
    for (int i = 0; i < numSmallFFTs; i++){
        smallInput.emplace_back(memManager.copyDataToDevice(smallCpuData));
        smallOutput.emplace_back(memManager.allocateMemoryOnDeviceComplex(smallShape));
    }
    for (int i = 0; i < num_iterations; i++){
        for (int j = 0; j < numSmallFFTs; ++j) {
            deconvBackend.forwardFFT(smallInput[j], smallOutput[j]);
        }
    }
    backend.sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto smallDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Small FFTs total time: " << smallDuration.count() << " mus" << std::endl;
    std::cout << "Average time per small FFT: " << (double)smallDuration.count() / numSmallFFTs << " mus" << std::endl;

    // Cleanup small FFT data
    for (int i = 0; i < numSmallFFTs; i++) {
        memManager.freeMemoryOnDevice(smallInput[i]);
        memManager.freeMemoryOnDevice(smallOutput[i]);
    }

    // Free CPU data
    cpuMemManager.freeMemoryOnDevice(smallCpuData);

    return smallDuration;
}

void runFFTPerformanceTest(const std::string& backendName) {
    std::cout << "=== FFT Performance Test for " << backendName << " Backend ===" << std::endl;

    try {
        // Get the backend manager and create backend
        IBackendManager& manager = BackendFactory::getInstance().getBackendManager(backendName);

        BackendConfig config;
        config.nThreads = 1;
        config.backendName = backendName;

        IBackend& backend = manager.getBackend(config);
        std::cout << "Backend device: " << backend.getDeviceString() << std::endl;

        // Get the CPU memory manager for host-side allocations
        IBackendMemoryManager& cpuMemManager = BackendFactory::getInstance().getDefaultBackendMemoryManager();

        {
            // Define total size for comparison
            // We want to compare: 1 big FFT vs 1000 smaller FFTs with same total size
            int num_iterations = 100;
            int size = 64;
            const int totalElements = size * size * size;  // Total volume
            int numSmallFFTs = 100;

            // Calculate size for small FFTs
            int smallFFTSize = totalElements / numSmallFFTs;
            // Ensure it's a perfect cube for 3D FFT
            int smallDim = std::round(std::cbrt(smallFFTSize));
            smallFFTSize = smallDim * smallDim * smallDim;

            std::cout << "Total elements: " << totalElements << std::endl;
            std::cout << "Number of small FFTs: " << numSmallFFTs << std::endl;
            std::cout << "Size per small FFT: " << smallFFTSize << " elements ("
                    << smallDim << "x" << smallDim << "x" << smallDim << ")" << std::endl;

            // Run large FFT test
            std::chrono::microseconds largeDuration{0};
            largeDuration = runLargeFFTTest(backendName, backend, cpuMemManager, totalElements, num_iterations);

            // Run small FFT test
            std::chrono::microseconds smallDuration{0};
            smallDuration = runSmallFFTTest(backendName, backend, cpuMemManager, smallDim, numSmallFFTs, num_iterations);

            // Calculate and display performance comparison
            std::cout << "\n--- Performance Comparison ---" << std::endl;
            std::cout << "large FFT time: " << largeDuration.count() << " mus" << std::endl;
            std::cout << "Small FFTs total time: " << smallDuration.count() << " mus" << std::endl;
            std::cout << "Time ratio (Small/large): " << (double)smallDuration.count() / largeDuration.count() << std::endl;

            if (smallDuration.count() < largeDuration.count()) {
                std::cout << "Small FFTs are " << (double)largeDuration.count() / smallDuration.count()
                        << "x faster than single large FFT" << std::endl;
            } else {
                std::cout << "large FFT is " << (double)smallDuration.count() / largeDuration.count()
                        << "x faster than small FFTs" << std::endl;
            }

            std::cout << "\n" << backendName << " FFT performance test completed successfully" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << backendName << " FFT performance test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CTEST_FULL_OUTPUT Starting FFT Performance Tests" << std::endl;

    Logging::init();

    // Test CPU backend
    runFFTPerformanceTest("cpu");

    std::cout << "\n" << std::string(50, '=') << "\n" << std::endl;

    // Test CUDA backend if available
    runFFTPerformanceTest("cuda");

    return 0;
}

