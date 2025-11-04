/*
Example demonstrating the lazy initialization multi-shape backend functionality

This example shows how to use the updated deconvolution backends
with lazy initialization, where FFT plans are created on-demand
when they're first needed for a specific shape.
*/

#include "backend/BackendFactory.h"
#include "backend/ComplexData.h"
#include <iostream>
#include <vector>

int main() {
    try {
        // Define multiple different shapes for processing
        std::vector<RectangleShape> shapes = {
            RectangleShape(128, 128, 64),   // Small volume
            RectangleShape(256, 256, 128),  // Medium volume  
            RectangleShape(512, 512, 256)   // Large volume
        };

        std::cout << "Creating CPU backend..." << std::endl;
        
        // Create CPU backend
        auto cpuBackend = BackendFactory::getInstance().createDeconvBackend("cpu");
        auto cpuMemManager = BackendFactory::getInstance().createMemManager("cpu");
        
        // Initialize backend - no FFT plans created yet (lazy initialization)
        cpuBackend->init();
        std::cout << "CPU backend initialized for lazy plan creation" << std::endl;
        
        // Test each shape - FFT plans will be created on-demand
        for (size_t i = 0; i < shapes.size(); ++i) {
            const auto& shape = shapes[i];
            std::cout << "Testing shape " << i << ": " 
                      << shape.width << "x" << shape.height << "x" << shape.depth << std::endl;
            
            // Allocate test data for this shape
            ComplexData inputData = cpuMemManager->allocateMemoryOnDevice(shape);
            ComplexData outputData = cpuMemManager->allocateMemoryOnDevice(shape);
            
            // Initialize with some test values
            for (int j = 0; j < shape.volume; ++j) {
                inputData.data[j][0] = static_cast<double>(j % 100) / 100.0;  // Real part
                inputData.data[j][1] = static_cast<double>((j * 2) % 100) / 100.0;  // Imaginary part
            }
            
            // Perform FFT operations - plans will be created lazily when needed
            std::cout << "  Performing forward FFT (plan will be created if needed)..." << std::endl;
            cpuBackend->forwardFFT(inputData, outputData);
            
            std::cout << "  Performing backward FFT (plan already exists)..." << std::endl;
            cpuBackend->backwardFFT(outputData, inputData);
            
            std::cout << "  FFT operations completed successfully for shape " << i << std::endl;
            
            // Clean up
            cpuMemManager->freeMemoryOnDevice(inputData);
            cpuMemManager->freeMemoryOnDevice(outputData);
        }
        
        // Clean up backend
        cpuBackend->cleanup();
        std::cout << "Backend cleanup completed" << std::endl;
        
        std::cout << "Lazy initialization multi-shape backend test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
