#include "../include/backend/Exceptions.h"
#include <iostream>

// Example function that demonstrates unified exception handling
void exampleMemoryOperation() {
    try {
        // Simulate a memory allocation that could fail
        size_t large_size = 1000000000000ULL; // 1TB - likely to fail
        void* ptr = malloc(large_size);
        
        // Use the unified memory allocation check macro
        MEMORY_ALLOC_CHECK(ptr, large_size, "CPU", "exampleMemoryOperation");
        
        // If we get here, allocation succeeded
        std::cout << "Memory allocation succeeded" << std::endl;
        free(ptr);
        
    } catch (const dolphin::backend::MemoryException& e) {
        // Catch memory-specific exceptions with detailed information
        std::cerr << "Caught memory exception: " << e.getDetailedMessage() << std::endl;
        
        // You can access specific properties
        if (e.getBackendType() == "CUDA") {
            std::cerr << "CUDA-specific error occurred" << std::endl;
        } else if (e.getBackendType() == "CPU") {
            std::cerr << "CPU-specific error occurred" << std::endl;
        }
        
        // You can also check the operation that failed
        if (e.getOperation() == "allocateMemoryOnDevice") {
            std::cerr << "Memory allocation failed" << std::endl;
        }
        
    } catch (const std::exception& e) {
        // Handle other types of exceptions
        std::cerr << "General exception: " << e.what() << std::endl;
    }
}