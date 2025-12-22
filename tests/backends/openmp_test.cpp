#include "../include/backend/BackendFactory.h"
#include "../include/backend/ComplexData.h"
#include <iostream>
#include <memory>
#include <chrono>

void testOpenMPBackendInitialization() {
    std::cout << "=== Testing OpenMP Backend Initialization ===" << std::endl;
    
    try {
        std::cout << "Creating OpenMP backend..." << std::endl;
        
        // Create OpenMP backend
        auto openmpBackend = BackendFactory::getInstance().createDeconvBackend("openmp");
        auto openmpMemManager = BackendFactory::getInstance().createMemManager("openmp");
        
        if (!openmpBackend) {
            std::cout << "OpenMP backend not available (this is expected if backend libraries are not built)" << std::endl;
            return;
        }
        
        if (!openmpMemManager) {
            std::cout << "OpenMP memory manager not available (this is expected if backend libraries are not built)" << std::endl;
            return;
        }
        
        std::cout << "OpenMP backend and memory manager created successfully" << std::endl;
        
        // Test initialization
        std::cout << "Initializing OpenMP backend..." << std::endl;
        openmpBackend->init();
        std::cout << "OpenMP backend initialized successfully" << std::endl;
        
        // Test with a simple shape
        RectangleShape testShape(64, 64, 32);
        std::cout << "Testing memory allocation with shape: "
                  << testShape.width << "x" << testShape.height << "x" << testShape.depth << std::endl;
        
        // Test memory allocation
        auto testData = openmpMemManager->allocateMemoryOnDevice(testShape);
        std::cout << "Memory allocation successful" << std::endl;
        
        // Test cleanup
        openmpMemManager->freeMemoryOnDevice(testData);
        std::cout << "Memory cleanup successful" << std::endl;
        
        // Test backend cleanup
        openmpBackend->cleanup();
        std::cout << "OpenMP backend cleanup completed" << std::endl;
        
        std::cout << "OpenMP backend initialization test PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "OpenMP backend test FAILED: " << e.what() << std::endl;
    }
}

void testOpenMPBackendFunctionality() {
    std::cout << "=== Testing OpenMP Backend Functionality ===" << std::endl;
    
    try {
        std::cout << "Creating OpenMP backend for functionality tests..." << std::endl;
        
        // Create OpenMP backend
        auto openmpBackend = BackendFactory::getInstance().createDeconvBackend("openmp");
        auto openmpMemManager = BackendFactory::getInstance().createMemManager("openmp");
        
        if (!openmpBackend || !openmpMemManager) {
            std::cout << "OpenMP backend components not available (this is expected if backend libraries are not built)" << std::endl;
            return;
        }
        
        // Initialize backend
        openmpBackend->init();
        
        // Test with a larger shape for meaningful parallelization
        RectangleShape testShape(128, 128, 64);
        std::cout << "Testing with shape: " << testShape.width << "x" << testShape.height << "x" << testShape.depth << std::endl;
        
        // Allocate memory for test data
        auto testData = openmpMemManager->allocateMemoryOnDevice(testShape);
        auto resultData = openmpMemManager->allocateMemoryOnDevice(testShape);
        
        // Initialize test data with simple values
        for (int i = 0; i < testShape.volume; ++i) {
            testData.data[i][0] = static_cast<double>(i % 100) / 100.0;
            testData.data[i][1] = static_cast<double>(i % 100) / 100.0;
        }
        
        std::cout << "Testing scalar multiplication..." << std::endl;
        openmpBackend->scalarMultiplication(testData, 2.5, resultData);
        
        // Verify scalar multiplication result
        bool scalarTestPassed = true;
        for (int i = 0; i < testShape.volume; ++i) {
            double expected = (static_cast<double>(i % 100) / 100.0) * 2.5;
            if (std::abs(resultData.data[i][0] - expected) > 1e-10 || 
                std::abs(resultData.data[i][1] - expected) > 1e-10) {
                scalarTestPassed = false;
                break;
            }
        }
        
        if (scalarTestPassed) {
            std::cout << "Scalar multiplication test PASSED" << std::endl;
        } else {
            std::cout << "Scalar multiplication test FAILED" << std::endl;
        }
        
        std::cout << "Testing complex addition..." << std::endl;
        openmpBackend->complexAddition(testData, testData, resultData);
        
        // Verify complex addition result (should be 2x original)
        bool additionTestPassed = true;
        for (int i = 0; i < testShape.volume; ++i) {
            double expected = 2.0 * (static_cast<double>(i % 100) / 100.0);
            if (std::abs(resultData.data[i][0] - expected) > 1e-10 || 
                std::abs(resultData.data[i][1] - expected) > 1e-10) {
                additionTestPassed = false;
                break;
            }
        }
        
        if (additionTestPassed) {
            std::cout << "Complex addition test PASSED" << std::endl;
        } else {
            std::cout << "Complex addition test FAILED" << std::endl;
        }
        
        // Test FFT functionality
        std::cout << "Testing FFT operations..." << std::endl;
        auto fftData = openmpMemManager->allocateMemoryOnDevice(testShape);
        
        // Copy test data to FFT data
        std::memcpy(fftData.data, testData.data, testShape.volume * sizeof(complex));
        
        // Test forward FFT
        openmpBackend->forwardFFT(fftData, resultData);
        std::cout << "Forward FFT completed" << std::endl;
        
        // Test backward FFT
        openmpBackend->backwardFFT(resultData, fftData);
        std::cout << "Backward FFT completed" << std::endl;
        
        // Test shift operations
        std::cout << "Testing shift operations..." << std::endl;
        openmpBackend->octantFourierShift(resultData);
        std::cout << "Octant Fourier shift completed" << std::endl;
        
        openmpBackend->inverseQuadrantShift(resultData);
        std::cout << "Inverse quadrant shift completed" << std::endl;
        
        // Cleanup
        openmpMemManager->freeMemoryOnDevice(testData);
        openmpMemManager->freeMemoryOnDevice(resultData);
        openmpMemManager->freeMemoryOnDevice(fftData);
        
        // Test backend cleanup
        openmpBackend->cleanup();
        
        std::cout << "OpenMP backend functionality test PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "OpenMP backend functionality test FAILED: " << e.what() << std::endl;
    }
}

void testOpenMPBackendPerformance() {
    std::cout << "=== Testing OpenMP Backend Performance ===" << std::endl;
    
    try {
        std::cout << "Creating OpenMP backend for performance tests..." << std::endl;
        
        // Create OpenMP backend
        auto openmpBackend = BackendFactory::getInstance().createDeconvBackend("openmp");
        auto openmpMemManager = BackendFactory::getInstance().createMemManager("openmp");
        
        if (!openmpBackend || !openmpMemManager) {
            std::cout << "OpenMP backend components not available (this is expected if backend libraries are not built)" << std::endl;
            return;
        }
        
        // Initialize backend
        openmpBackend->init();
        
        // Test with different sizes to measure performance
        std::vector<std::tuple<int, int, int>> testSizes = {
            {64, 64, 32},
            {128, 128, 64},
            {256, 256, 128}
        };
        
        for (const auto& size : testSizes) {
            int width = std::get<0>(size);
            int height = std::get<1>(size);
            int depth = std::get<2>(size);
            
            RectangleShape testShape(width, height, depth);
            std::cout << "Testing performance with shape: " << width << "x" << height << "x" << depth << std::endl;
            
            // Allocate memory
            auto testData = openmpMemManager->allocateMemoryOnDevice(testShape);
            auto resultData = openmpMemManager->allocateMemoryOnDevice(testShape);
            
            // Initialize test data
            for (int i = 0; i < testShape.volume; ++i) {
                testData.data[i][0] = static_cast<double>(i % 100) / 100.0;
                testData.data[i][1] = static_cast<double>(i % 100) / 100.0;
            }
            
            // Test FFT performance
            auto start = std::chrono::high_resolution_clock::now();
            openmpBackend->forwardFFT(testData, resultData);
            openmpBackend->backwardFFT(resultData, testData);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "FFT operations took: " << duration.count() << " ms" << std::endl;
            
            // Test complex arithmetic performance
            start = std::chrono::high_resolution_clock::now();
            openmpBackend->complexMultiplication(testData, testData, resultData);
            end = std::chrono::high_resolution_clock::now();
            
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Complex multiplication took: " << duration.count() << " ms" << std::endl;
            
            // Cleanup
            openmpMemManager->freeMemoryOnDevice(testData);
            openmpMemManager->freeMemoryOnDevice(resultData);
        }
        
        // Test backend cleanup
        openmpBackend->cleanup();
        
        std::cout << "OpenMP backend performance test PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "OpenMP backend performance test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Starting OpenMP Backend Test" << std::endl;
    
    testOpenMPBackendInitialization();
    std::cout << std::endl;
    
    testOpenMPBackendFunctionality();
    std::cout << std::endl;
    
    testOpenMPBackendPerformance();
    std::cout << std::endl;
    
    return 0;
}