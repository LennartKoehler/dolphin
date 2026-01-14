#include "backend/BackendFactory.h"
#include "dolphinbackend/ComplexData.h"
#include <iostream>
#include <memory>

void testCPUBackendInitialization() {
    std::cout << "=== Testing CPU Backend Initialization ===" << std::endl;
    
    try {
        std::cout << "Creating CPU backend..." << std::endl;
        
        // Create CPU backend
        auto cpuBackend = BackendFactory::getInstance().createDeconvBackend("cpu");
        auto cpuMemManager = BackendFactory::getInstance().createMemManager("cpu");
        
        if (!cpuBackend) {
            std::cout << "CPU backend not available (this is expected if backend libraries are not built)" << std::endl;
            return;
        }
        
        if (!cpuMemManager) {
            std::cout << "CPU memory manager not available (this is expected if backend libraries are not built)" << std::endl;
            return;
        }
        
        std::cout << "CPU backend and memory manager created successfully" << std::endl;
        
        // Test initialization
        std::cout << "Initializing CPU backend..." << std::endl;
        cpuBackend->init();
        std::cout << "CPU backend initialized successfully" << std::endl;
        
        // Test with a simple shape
        RectangleShape testShape(64, 64, 32);
        std::cout << "Testing memory allocation with shape: "
                  << testShape.width << "x" << testShape.height << "x" << testShape.depth << std::endl;
        
        // Test memory allocation
        auto testData = cpuMemManager->allocateMemoryOnDevice(testShape);
        std::cout << "Memory allocation successful" << std::endl;
        
        // Test cleanup
        cpuMemManager->freeMemoryOnDevice(testData);
        std::cout << "Memory cleanup successful" << std::endl;
        
        // Test backend cleanup
        cpuBackend->cleanup();
        std::cout << "CPU backend cleanup completed" << std::endl;
        
        std::cout << "CPU backend initialization test PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "CPU backend test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Starting CPU Backend Test" << std::endl;
    testCPUBackendInitialization();
    return 0;
}