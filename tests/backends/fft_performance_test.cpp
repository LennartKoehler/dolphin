#include "../include/backend/BackendFactory.h"
#include "../include/backend/ComplexData.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <vector>

ComplexData createAndFillComplexDataWithRandom(const RectangleShape& shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dist(-1.0, 1.0);
    std::uniform_real_distribution<> imag_dist(-1.0, 1.0);
    
    // Create CPU backend for data creation
    auto cpuMemManager = BackendFactory::getInstance().createMemManager("cpu");
    if (!cpuMemManager) {
        throw std::runtime_error("Failed to create CPU memory manager");
    }
    
    // Allocate data using CPU backend
    ComplexData result = cpuMemManager->allocateMemoryOnDevice(shape);
    
    // Fill with random data
    for (int i = 0; i < result.size.volume; ++i) {
        result.data[i][0] = real_dist(gen);  // Real part
        result.data[i][1] = imag_dist(gen);  // Imaginary part
    }
    
    return result;
}

void runFFTPerformanceTest(const std::string& backendName) {
    std::cout << "=== FFT Performance Test for " << backendName << " Backend ===" << std::endl;
    
    try {
        // Create backend and memory manager
        auto backend = BackendFactory::getInstance().createDeconvBackend(backendName);
        auto memManager = BackendFactory::getInstance().createMemManager(backendName);
        
        if (!backend || !memManager) {
            std::cout << backendName << " backend not available" << std::endl;
            return;
        }
        
        // Initialize backend
        backend->init();
        
        // Define total size for comparison
        // We want to compare: 1 big FFT vs 1000 smaller FFTs with same total size
        int num_iterations = 300;
        int size = 128;
        const int totalElements = size * size * size;  // Total volume
        const int numSmallFFTs = 1000;
        
        // Calculate size for small FFTs
        int smallFFTSize = totalElements / numSmallFFTs;
        // Ensure it's a perfect cube for 3D FFT
        int smallDim = std::round(std::cbrt(smallFFTSize));
        smallFFTSize = smallDim * smallDim * smallDim;
        
        std::cout << "Total elements: " << totalElements << std::endl;
        std::cout << "Number of small FFTs: " << numSmallFFTs << std::endl;
        std::cout << "Size per small FFT: " << smallFFTSize << " elements (" 
                  << smallDim << "x" << smallDim << "x" << smallDim << ")" << std::endl;
        
        // Test 1: One large FFT
        std::cout << "\n--- Test 1: One large FFT ---" << std::endl;
        
        // Calculate dimensions for large FFT
        int largeDim = std::round(std::cbrt(totalElements));
        RectangleShape largeShape(largeDim, largeDim, largeDim);
        std::cout << "large FFT shape: " << largeShape.width << "x" << largeShape.height << "x" << largeShape.depth << std::endl;
        
        // Create and fill large data on CPU, then copy to test backend
        ComplexData largeCpuData = createAndFillComplexDataWithRandom(largeShape);
        auto largeInput = memManager->copyDataToDevice(largeCpuData);
        auto largeOutput = memManager->allocateMemoryOnDevice(largeShape);
        
        // Free CPU data after copying to device (using CPU backend to free it)
        auto cpuMemManager = BackendFactory::getInstance().createMemManager("cpu");
        cpuMemManager->freeMemoryOnDevice(largeCpuData);
        
        // Time the large FFT
        backend->initializePlan(largeShape);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++){
            backend->forwardFFT(largeInput, largeOutput);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto largeDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "large FFT time: " << largeDuration.count() << " mus" << std::endl;
        
        // Cleanup large data
        memManager->freeMemoryOnDevice(largeInput);
        memManager->freeMemoryOnDevice(largeOutput);
        
        // Test 2: 1000 smaller FFTs
        std::cout << "\n--- Test 2: " << numSmallFFTs << " Smaller FFTs ---" << std::endl;
        
        RectangleShape smallShape(smallDim, smallDim, smallDim);
        std::cout << "Small FFT shape: " << smallShape.width << "x" << smallShape.height << "x" << smallShape.depth << std::endl;
        
        // Create and fill small data on CPU, then copy to test backend
        ComplexData smallCpuData = createAndFillComplexDataWithRandom(smallShape);
        std::vector<ComplexData> smallInput;
        std::vector<ComplexData> smallOutput;
        

       

        backend->initializePlan(smallShape);


        // Time the small FFTs
        start = std::chrono::high_resolution_clock::now();
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
        // Free CPU data after copying to device (using CPU backend to free it)
        cpuMemManager->freeMemoryOnDevice(smallCpuData);
        end = std::chrono::high_resolution_clock::now();
        auto smallDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Small FFTs total time: " << smallDuration.count() << " mus" << std::endl;
        std::cout << "Average time per small FFT: " << (double)smallDuration.count() / numSmallFFTs << " mus" << std::endl;
        
        
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
        
        // Cleanup small FFT data
        for (int i = 0; i < numSmallFFTs; i++) {
            memManager->freeMemoryOnDevice(smallInput[i]);
            memManager->freeMemoryOnDevice(smallOutput[i]);
        }
        
        // Cleanup backend
        backend->cleanup();
        
        std::cout << "\n" << backendName << " FFT performance test completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << backendName << " FFT performance test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Starting FFT Performance Tests" << std::endl;
    
    // Test CPU backend
    runFFTPerformanceTest("cpu");
    
    std::cout << "\n" << std::string(50, '=') << "\n" << std::endl;
    
    // Test CUDA backend if available
    runFFTPerformanceTest("cuda");
    
    return 0;
}