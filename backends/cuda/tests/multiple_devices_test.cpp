#include <iostream>
#include <memory>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include <thread>
#include "dolphinbackend/IBackend.h"
#include "CUDABackend.h"
void testMultipleDevices() {
    std::cout << "=== Testing CUDA Backend Initialization ===" << std::endl;
    
    try {
        std::cout << "Creating CUDA backend..." << std::endl;
        // Create CUDA backend

        std::shared_ptr<IBackend> backend = std::shared_ptr<CUDABackend>(CUDABackend::create());
        IBackendMemoryManager* cudaMemManager = backend->getMemoryManagerPtr();
        IDeconvolutionBackend& cudaBackendtemp = backend->mutableDeconvManager();
        IDeconvolutionBackend* cudaBackend = &cudaBackendtemp;
        std::cout << "Number of devices detectes" << backend->getNumberDevices() << std::endl;
        if (!cudaBackend) {
            std::cout << "CUDA backend not available (this is expected if CUDA libraries are not built)" << std::endl;
            return;
        }
        
        backend->clone();

        std::thread testThread = std::thread([backend](){
            backend->clone();
        });
        std::thread testThread2([backend](){
            backend->clone();
        });


        if (!cudaMemManager) {
            std::cout << "CUDA memory manager not available (this is expected if CUDA libraries are not built)" << std::endl;
            return;
        }
        
        std::cout << "CUDA backend and memory manager created successfully" << std::endl;
        
        // Test initialization
        std::cout << "Initializing CUDA backend..." << std::endl;
        cudaBackend->init();
        std::cout << "CUDA backend initialized successfully" << std::endl;
        
        // Test with a simple shape
        CuboidShape testShape(64, 64, 32);
        std::cout << "Testing memory allocation with shape: "
                  << testShape.width << "x" << testShape.height << "x" << testShape.depth << std::endl;
        
        // Test memory allocation
        auto testData = cudaMemManager->allocateMemoryOnDevice(testShape);
        std::cout << "Memory allocation successful" << std::endl;
        
        // Test cleanup
        cudaMemManager->freeMemoryOnDevice(testData);
        std::cout << "Memory cleanup successful" << std::endl;
        
        // Test backend cleanup
        cudaBackend->cleanup();
        std::cout << "CUDA backend cleanup completed" << std::endl;
        
        std::cout << "CUDA backend initialization test PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "CUDA backend test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Starting CUDA Backend Test" << std::endl;
    testMultipleDevices();
    return 0;
}