#include <dolphin/backend/BackendFactory.h>
#include <dolphinbackend/ComplexData.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <vector>


void fillComplexDataWithRandom(const ComplexData& result) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dist(-1.0, 1.0);
    std::uniform_real_distribution<> imag_dist(-1.0, 1.0);

    
    // Fill with random data
    for (int i = 0; i < result.size.volume; ++i) {
        result.data[i][0] = real_dist(gen);  // Real part
        result.data[i][1] = imag_dist(gen);  // Imaginary part
    }

}

std::chrono::microseconds runLargeFFTTest(
    const std::string& backendName,
    IDeconvolutionBackend* backend,
    IBackendMemoryManager* memManager,
    IBackendMemoryManager* cpuMemManager,
    int totalElements,
    int num_iterations
) {
    std::cout << "\n--- Test 1: One large FFT ---" << std::endl;
    
    // Calculate dimensions for large FFT
    int largeDim = std::round(std::cbrt(totalElements));
    RectangleShape largeShape(largeDim, largeDim, largeDim);
    std::cout << "large FFT shape: " << largeShape.width << "x" << largeShape.height << "x" << largeShape.depth << std::endl;
    
    // Create and fill large data on CPU, then copy to test backend
    ComplexData largeCpuData = cpuMemManager->allocateMemoryOnDevice(largeShape);
    fillComplexDataWithRandom(largeCpuData);
    auto largeInput = memManager->copyDataToDevice(largeCpuData);
    auto largeOutput = memManager->allocateMemoryOnDevice(largeShape);
    
    // Free CPU data after copying to device (using CPU backend to free it)
    cpuMemManager->freeMemoryOnDevice(largeCpuData);
    
    // Time the large FFT
    backend->initializePlan(largeShape);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++){
        backend->forwardFFT(largeInput, largeOutput);
    }
    backend->sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto largeDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "large FFT time: " << largeDuration.count() << " mus" << std::endl;
    
    // Cleanup large data
    memManager->freeMemoryOnDevice(largeInput);
    memManager->freeMemoryOnDevice(largeOutput);
    
    return largeDuration;
}

std::chrono::microseconds runSmallFFTTest(
    const std::string& backendName,
    IDeconvolutionBackend* backend,
    IBackendMemoryManager* memManager,
    IBackendMemoryManager* cpuMemManager,
    int smallDim,
    int numSmallFFTs,
    int num_iterations
) {
    std::cout << "\n--- Test 2: " << numSmallFFTs << " Smaller FFTs ---" << std::endl;
    
    RectangleShape smallShape(smallDim, smallDim, smallDim);
    std::cout << "Small FFT shape: " << smallShape.width << "x" << smallShape.height << "x" << smallShape.depth << std::endl;
    
    // Create and fill small data on CPU, then copy to test backend
    ComplexData smallCpuData = cpuMemManager->allocateMemoryOnDevice(smallShape);
    fillComplexDataWithRandom(smallCpuData);
    std::vector<ComplexData> smallInput;
    std::vector<ComplexData> smallOutput;
    
    backend->cleanup();
    backend->initializePlan(smallShape);

    // Time the small FFTs
    auto start = std::chrono::high_resolution_clock::now();
    // Reserve space to avoid reallocations
    smallInput.reserve(numSmallFFTs);
    smallOutput.reserve(numSmallFFTs);
    
    // Create input and output data for each small FFT
    for (int i = 0; i < numSmallFFTs; i++){
        smallInput.emplace_back(memManager->copyDataToDevice(smallCpuData));
        smallOutput.emplace_back(memManager->allocateMemoryOnDevice(smallShape));
    } 
    for (int i = 0; i < num_iterations; i++){
        for (int j = 0; j < numSmallFFTs; ++j) {
            backend->forwardFFT(smallInput[j], smallOutput[j]);
        }
    }
    backend->sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto smallDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Small FFTs total time: " << smallDuration.count() << " mus" << std::endl;
    std::cout << "Average time per small FFT: " << (double)smallDuration.count() / numSmallFFTs << " mus" << std::endl;
    
    // Cleanup small FFT data
    for (int i = 0; i < numSmallFFTs; i++) {
        memManager->freeMemoryOnDevice(smallInput[i]);
        memManager->freeMemoryOnDevice(smallOutput[i]);
    }
    
    // Free CPU data
    cpuMemManager->freeMemoryOnDevice(smallCpuData);
    
    return smallDuration;
}

void runFFTPerformanceTest(const std::string& backendName) {
    std::cout << "=== FFT Performance Test for " << backendName << " Backend ===" << std::endl;
    
    try {
        // Create backend and memory manager
        auto backend = BackendFactory::getInstance().createDeconvBackend(backendName);
        auto memManager = BackendFactory::getInstance().createMemManager(backendName);
        auto cpuMemManager = BackendFactory::getInstance().createMemManager("cpu");

        { 
            if (!backend || !memManager) {
                std::cout << backendName << " backend not available" << std::endl;
                return;
            }
            
            // Initialize backend
            backend->init();
            
            // Define total size for comparison
            // We want to compare: 1 big FFT vs 1000 smaller FFTs with same total size
            int num_iterations = 100;
            int size = 64;
            const int totalElements = size * size * size;  // Total volume
            int numSmallFFTs = 100;

            // Calculate size for small FFTs
            int smallFFTSize = totalElements / numSmallFFTs;
            // numSmallFFTs = 1; //TESTVALUE
            // Ensure it's a perfect cube for 3D FFT
            int smallDim = std::round(std::cbrt(smallFFTSize));
            // smallDim = 67; //TESTVALUE
            smallFFTSize = smallDim * smallDim * smallDim;
            
            std::cout << "Total elements: " << totalElements << std::endl;
            std::cout << "Number of small FFTs: " << numSmallFFTs << std::endl;
            std::cout << "Size per small FFT: " << smallFFTSize << " elements (" 
                    << smallDim << "x" << smallDim << "x" << smallDim << ")" << std::endl;
            
            // Run large FFT test
            std::chrono::microseconds largeDuration{0}; 
            largeDuration = runLargeFFTTest(backendName, backend.get(), memManager.get(), cpuMemManager.get(), totalElements, num_iterations);
            
            // Run small FFT test
            std::chrono::microseconds smallDuration{0};
            smallDuration = runSmallFFTTest(backendName, backend.get(), memManager.get(), cpuMemManager.get(), smallDim, numSmallFFTs, num_iterations);
            
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
            
            // Cleanup backend
            backend->cleanup();
            
            std::cout << "\n" << backendName << " FFT performance test completed successfully" << std::endl;
        } 
    } catch (const std::exception& e) {
        std::cerr << backendName << " FFT performance test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CTEST_FULL_OUTPUT Starting FFT Performance Tests" << std::endl;
    
    // Test CPU backend
    runFFTPerformanceTest("cpu");
    
    std::cout << "\n" << std::string(50, '=') << "\n" << std::endl;
    
    // Test CUDA backend if available
    runFFTPerformanceTest("cuda");
    
    return 0;
}