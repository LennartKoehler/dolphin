#include "CUDABackendManager.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// Example function that uses a CUDA backend with thread-specific stream
void processWithCUDABackend(int threadId) {
    try {
        std::cout << "Thread " << threadId << " (ID: " << std::this_thread::get_id()
                  << "): Acquiring CUDA backend..." << std::endl;
        
        // Get backend for current thread (will create dedicated CUDA stream)
        auto backend = dolphin::backend::CUDADeconvolutionBackend::getBackendForCurrentThread();
        
        std::cout << "Thread " << threadId << ": Acquired CUDA backend successfully" << std::endl;
        
        // Initialize the backend for this thread
        backend->initThread();
        
        // Simulate some work with CUDA operations
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        std::cout << "Thread " << threadId << ": Releasing CUDA backend..." << std::endl;
        
        // Release backend back to manager (only current thread can release it)
        dolphin::backend::CUDADeconvolutionBackend::releaseBackendForCurrentThread(std::move(backend));
        
        std::cout << "Thread " << threadId << ": Released CUDA backend successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Thread " << threadId << ": Error - " << e.what() << std::endl;
    }
}

int main() {
    try {
        // Initialize the CUDA backend manager with support for 4 threads
        auto& manager = dolphin::backend::CUDABackendManager::getInstance();
        manager.setMaxThreads(4);
        
        std::cout << "CUDA Backend Manager initialized with support for "
                  << manager.getMaxThreads() << " threads" << std::endl;
        
        // Create multiple threads to test thread-specific backend management
        std::vector<std::thread> threads;
        const int numThreads = 6; // This will exceed the default maxThreads to show error handling
        
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(processWithCUDABackend, i);
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        std::cout << "All threads completed. Active threads: "
                  << manager.getActiveThreads() << ", Total backends: "
                  << manager.getTotalBackends() << std::endl;
        
        // Clean up
        manager.cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "Main error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}