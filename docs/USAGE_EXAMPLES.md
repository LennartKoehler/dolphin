# DOLPHIN Usage Examples: CPU/GPU Architecture

This document provides comprehensive usage examples for the refactored DOLPHIN CPU/GPU architecture, demonstrating how to effectively use both backends for different scenarios and optimization levels.

## Table of Contents
- [Basic CPU Usage](#basic-cpu-usage)
- [Basic GPU Usage](#basic-gpu-usage)
- [Advanced Configuration Examples](#advanced-configuration-examples)
- [Performance Monitoring Examples](#performance-monitoring-examples)
- [Error Handling Examples](#error-handling-examples)
- [Migrating Legacy Configurations](#migrating-legacy-configurations)
- [Benchmarking and Optimization](#benchmarking-and-optimization)

## Basic CPU Usage

### Simple CPU Deconvolution

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h"
#include <iostream>

int main() {
    try {
        // Create Dolphin instance
        Dolphin dolphin;
        
        // Create CPU-based Richardson-Lucy algorithm
        std::unique_ptr<RLDeconvolutionAlgorithm> algorithm = std::make_unique<RLDeconvolutionAlgorithm>();
        
        // Configure algorithm for CPU processing
        DeconvolutionConfig config;
        config.algorithmName = "rl";
        config.iterations = 50;
        config.epsilon = 1e-6;
        config.time = true;           // Enable timing
        config.grid = false;          // Disable grid processing for small images
        config.gpu = "none";          // Explicitly use CPU
        
        algorithm->configure(config);
        
        // Set up deconvolution request
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = "input.tif";
        request.setupConfig->psfFilePath = "psf.tif";
        request.setupConfig->gpu = "none";
        request.setupConfig->time = true;
        
        // Run deconvolution
        DeconvolutionResult result = dolphin.deconvolve(request);
        
        // Save result
        result.hyperstack.saveAsTifFile("result_cpu.tif");
        
        std::cout << "CPU deconvolution completed successfully" << std::endl;
        std::cout << "Processing time: " << result.processingTime << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Using Algorithm Factory

```cpp
#include "Dolphin.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include <iostream>

int main() {
    try {
        // Create Dolphin instance
        Dolphin dolphin;
        
        // Create algorithm instance through factory
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        auto algorithm = factory.create("rltv", "none");  // RLTV algorithm, CPU backend
        
        if (!algorithm) {
            std::cerr << "Failed to create algorithm" << std::endl;
            return 1;
        }
        
        // Configure through SetupConfig (recommended for complex scenarios)
        auto setupConfig = std::make_shared<SetupConfig>();
        setupConfig->imagePath = "large_input.tif";
        setupConfig->psfFilePath = "psf.tif";
        setupConfig->gpu = "none";
        
        // Configure algorithm parameters
        setupConfig->deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
        setupConfig->deconvolutionConfig->algorithmName = "rltv";
        setupConfig->deconvolutionConfig->iterations = 75;
        setupConfig->deconvolutionConfig->lambda = 0.01;
        setupConfig->deconvolutionConfig->grid = true;
        setupConfig->deconvolutionConfig->subimageSize = 256;  // Optimized for CPU
        setupConfig->deconvolutionConfig->time = true;
        
        // Set up and run deconvolution
        DeconvolutionRequest request;
        request.setupConfig = setupConfig;
        request.save_separate = false;
        
        DeconvolutionResult result = dolphin.deconvolve(request);
        
        // Handle result with automatic backend selection
        if (result.success) {
            std::cout << "CPU processing completed successfully" << std::endl;
            result.hyperstack.saveAsTifFile("result_cpu_optimized.tif");
        }
        
        // Cleanup factory instance (recommended for proper cleanup)
        factory.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### CLI-Style Processing

```cpp
#include "Dolphin.h"
#include <iostream>
#include <chrono>

int main() {
    try {
        // Simulate CLI-style processing
        Dolph dolphin;
        
        // Parse and process command line arguments (simplified)
        std::string imagePath = "input.tif";
        std::string psfPath = "psf.tif";
        std::string algorithm = "rl";
        bool useGPU = false;
        int iterations = 50;
        
        // Create algorithm with auto-detection
        auto factory = DeconfolutionAlgorithmFactory::getInstance();
        auto algorithm = factory.create(algorithm, useGPU ? "cuda" : "none");
        
        // Set up configuration
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = imagePath;
        request.setupConfig->psfFilePath = psfPath;
        request.setupConfig->gpu = useGPU ? "cuda" : "none";
        request.setupConfig->time = true;
        request.setupConfig->sep = false;
        
        request.setupConfig->deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
        request.setupConfig->deconvolutionConfig->algorithmName = algorithm;
        request.setupConfig->deconvolutionConfig->iterations = iterations;
        request.setupConfig->deconvolutionConfig->grid = true;
        request.setupConfig->deconvolutionConfig->subimageSize = 0;  // Auto adjust
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute deconvolution
        DeconvolutionResult result = dolphin.deconvolve(request);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (result.success) {
            std::cout << "Processing completed in " << duration.count() << " ms" << std::endl;
            std::cout << "Result saved to: result.tif" << std::endl;
        } else {
            std::cerr << "Processing failed: " << result.errorMessage << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Critical error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Basic GPU Usage

### Simple GPU Deconvolution

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#include <iostream>

int main() {
    try {
        // Check CUDA availability
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        if (!factory.isGPUSupported()) {
            std::cout << "CUDA not available, falling back to CPU" << std::endl;
            return runCPUExample();  // Assume CPU example function exists
        }
        
        // Create Dolphin instance
        Dolphin dolphin;
        
        // Create GPU-based Richardson-Lucy algorithm
        std::unique_ptr<RLDeconvolutionAlgorithm> algorithm = std::make_unique<RLDeconvolutionAlgorithm>();
        
        // Configure algorithm for GPU processing
        DeconvolutionConfig config;
        config.algorithmName = "rl";
        config.iterations = 100;
        config.epsilon = 1e-6;
        config.time = true;           // Enable timing
        config.grid = true;           // Enable grid processing (benefits GPU)
        config.subimageSize = 512;    // Larger subimages benefit GPU
        config.gpu = "cuda";          // Enable GPU backend
        
        algorithm->configure(config);
        
        // Cast to GPU backend for device management
        auto* gpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmGPU*>(algorithm.get());
        if (gpuAlgorithm) {
            // List available GPUs
            auto availableGPUs = gpuAlgorithm->getAvailableGPUs();
            std::cout << "Found " << availableGPUs.size() << " GPU device(s)" << std::endl;
            
            // Use optimal GPU (usually device 0)
            if (!availableGPUs.empty()) {
                gpuAlgorithm->setGPUDevice(availableGPUs[0]);
                std::cout << "Using GPU device " << availableGPUs[0] << std::endl;
            }
            
            // Start performance monitoring
            gpuAlgorithm->resetGPUStats();
        }
        
        // Set up deconvolution request
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = "large_input.tif";
        request.setupConfig->psfFilePath = "psf.tif";
        request.setupConfig->gpu = "cuda";
        request.setupConfig->time = true;
        
        // Run deconvolution
        auto start = std::chrono::high_resolution_clock::now();
        DeconvolutionResult result = dolphin.deconvolve(request);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Print GPU performance metrics
        if (gpuAlgorithm) {
            auto runtimes = gpuAlgorithm->getGPURuntimeHistory();
            if (!runtimes.empty()) {
                double avgRuntime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
                std::cout << "Average GPU processing time: " << avgRuntime << " ms" << std::endl;
                std::cout << "Peak GPU memory usage: " << gpuAlgorithm->getPeakGPUMemory() / (1024.0 * 1024.0) << " MB" << std::endl;
            }
        }
        
        // Save result
        result.hyperstack.saveAsTifFile("result_gpu.tif");
        
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "GPU deconvolution completed successfully" << std::endl;
        std::cout << "Total processing time: " << totalDuration.count() << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Advanced GPU Device Management

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#include <iostream>

class GPUManager {
public:
    GPUManager() {
        // Check GPU availability
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        if (factory.isGPUSupported()) {
            m_gpuAvailable = true;
            m_deviceCount = getAvailableGPUCount();
            // Initialize GPU device information
            initializeGPUInfo();
        }
    }
    
    bool isGPUAvailable() const { return m_gpuAvailable; }
    int getDeviceCount() const { return m_deviceCount; }
    
    /**
     * Select optimal GPU based on various criteria
     * @return Best GPU device ID, or -1 if no suitable GPU found
     */
    int selectOptimalGPU() const {
        if (!m_gpuAvailable || m_deviceCount == 0) {
            return -1;
        }
        
        int bestDevice = 0;
        size_t bestScore = 0;
        
        for (int device : m_availableDevices) {
            size_t score = calculateGPUScore(device);
            std::cout << "GPU " << device << " score: " << score 
                      << " (Memory: " << getDeviceMemory(device) / (1024.0 * 1024.0) << " MB, "
                      << "Compute: " << getComputeCapability(device) << ")" << std::endl;
            
            if (score > bestScore) {
                bestScore = score;
                bestDevice = device;
            }
        }
        
        return bestDevice;
    }
    
    /**
     * Print detailed GPU configuration information
     */
    void printGPUConfiguration() const {
        if (!m_gpuAvailable) {
            std::cout << "No GPU available" << std::endl;
            return;
        }
        
        std::cout << "GPU Configuration:" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << "Available Device Count: " << m_deviceCount << std::endl;
        
        for (int device : m_availableDevices) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            
            std::cout << "\nDevice " << device << ":" << std::endl;
            std::cout << "  Name: " << props.name << std::endl;
            std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
            std::cout << "  Memory: " << (props.totalGlobalMem / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  Clock: " << (props.clockRate / 1000.0) << " MHz" << std::endl;
            std::cout << "  Multiprocessors: " << props.multiProcessorCount << std::endl;
            std::cout << "  Max Threads/Block: " << props.maxThreadsPerBlock << std::endl;
        }
    }
    
private:
    bool m_gpuAvailable;
    int m_deviceCount;
    std::vector<int> m_availableDevices;
    
    void initializeGPUInfo() {
        m_availableDevices.clear();
        for (int i = 0; i < 10; i++) {  // Reasonable limit
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, i);
            if (err == cudaSuccess) {
                m_availableDevices.push_back(i);
            }
        }
    }
    
    size_t calculateGPUScore(int device) const {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        // Score based on memory, compute capability, and multiprocessor count
        size_t score = props.totalGlobalMem;             // Weight memory heavily
        score += props.multiProcessorCount * 1000000;     // Weight multiprocessors
        score *= (props.major * 10 + props.minor);        // Weight compute capability
        
        return score;
    }
    
    size_t getDeviceMemory(int device) const {
        cudaDeviceProp props;
        cudaError_t err = cudaGetDeviceProperties(&props, device);
        return (err == cudaSuccess) ? props.totalGlobalMem : 0;
    }
    
    std::string getComputeCapability(int device) const {
        cudaDeviceProp props;
        cudaError_t err = cudaGetDeviceProperties(&props, device);
        if (err == cudaSuccess) {
            return std::to_string(props.major) + "." + std::to_string(props.minor);
        }
        return "N/A";
    }
    
    int getAvailableGPUCount() const {
        int count = 0;
        for (int i = 0; i < 10; i++) {  // Reasonable limit
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, i);
            if (err == cudaSuccess) {
                count++;
            }
        }
        return count;
    }
};

int main() {
    GPUManager gpuManager;
    
    // Print GPU configuration
    gpuManager.printGPUConfiguration();
    
    try {
        Dolphin dolphin;
        
        // Use GPU if available, otherwise fall back to CPU
        if (gpuManager.isGPUAvailable()) {
            int optimalGPU = gpuManager.selectOptimalGPU();
            std::cout << "\nOptimal GPU selected: " << optimalGPU << std::endl;
            
            // Create GPU algorithm
            std::unique_ptr<RLDeconvolutionAlgorithm> algorithm = std::make_unique<RLDeconvolutionAlgorithm>();
            auto* gpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmGPU*>(algorithm.get());
            
            // Configure for GPU processing
            DeconvolutionConfig config;
            config.algorithmName = "rltv";
            config.iterations = 150;
            config.epsilon = 1e-6;
            config.time = true;
            config.grid = true;
            config.subimageSize = 512;    // Optimal for GPU
            config.gpu = "cuda";
            config.usePinnedMemory = true;  // Enable for better performance
            config.useAsyncTransfers = true; // Enable for better performance
            config.useCUBEKernels = true;     // Use optimized kernels
            
            algorithm->configure(config);
            
            // Set optimal GPU device
            if (gpuAlgorithm) {
                gpuAlgorithm->setGPUDevice(optimalGPU);
                gpuAlgorithm->resetGPUStats();
            }
            
            // Additional GPU-specific configuration
            auto setupConfig = std::make_shared<SetupConfig>();
            setupConfig->imagePath = "very_large_input.tif";
            setupConfig->psfFilePath = "psf.tif";
            setupConfig->gpu = "cuda";
            
            DeconvolutionRequest request;
            request.setupConfig = setupConfig;
            request.save_subimages = false;
            
            // Run GPU deconvolution
            auto start = std::chrono::high_resolution_clock::now();
            DeconvolutionResult result = dolphin.deconvolve(request);
            auto end = std::chrono::high_resolution_clock::now();
            
            // Print GPU-specific metrics
            if (gpuAlgorithm) {
                gpuAlgorithm->logPerformanceMetrics();
                gpuAlgorithm->logMemoryUsage("final_report");
            }
            
            auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "GPU processing completed in " << totalDuration.count() << " ms" << std::endl;
            
        } else {
            std::cout << "No GPU available, using CPU processing" << std::endl;
            // Run CPU version (implementation would be similar to CPU example)
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Dynamic Backend Selection

```cpp
#include "Dolphin.h"
#include <iostream>
#include <stdexcept>

class DeconvolutionProcessor {
public:
    enum BackendType { AUTO, CPU, GPU };
    
    DeconvolutionProcessor(BackendType type = AUTO) : m_backendType(type) {
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        m_gpuAvailable = factory.isGPUSupported();
        
        if (m_backendType == GPU && !m_gpuAvailable) {
            std::cout << "GPU requested but not available, switching to CPU" << std::endl;
            m_backendType = CPU;
        }
    }
    
    DeconvolutionResult processImage(const std::string& imagePath, 
                                   const std::string& psfPath,
                                   const std::string& algorithmName = "rl",
                                   int iterations = 100) {
        
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        std::string backend = (m_backendType == GPU) ? "cuda" : "none";
        
        auto algorithm = factory.create(algorithmName, backend);
        if (!algorithm) {
            throw std::runtime_error("Failed to create algorithm");
        }
        
        // Configure based on backend type
        configureAlgorithm(*algorithm, backend, iterations);
        
        // Set up request
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = imagePath;
        request.setupConfig->psfFilePath = psfPath;
        request.setupConfig->gpu = backend;
        
        // Apply backend-specific optimizations
        configureBackendSpecific(request, backend);
        
        // Execute and return results
        return Dolphin().deconvolve(request);
    }
    
    bool isGPUAvailable() const { return m_gpuAvailable; }
    
private:
    BackendType m_backendType;
    bool m_gpuAvailable;
    
    void configureAlgorithm(BaseDeconvolutionAlgorithm& algorithm, 
                          const std::string& backend, 
                          int iterations) {
        
        DeconvolutionConfig config;
        config.algorithmName = algorithm.getAlgorithmName();  // Assume this method exists
        config.iterations = iterations;
        config.epsilon = 1e-6;
        config.time = true;
        config.grid = (backend == "cuda");  // Enable grid for GPU
        
        if (backend == "cuda") {
            config.subimageSize = 512;            // Larger for GPU
            config.usePinnedMemory = true;       // Optimize transfers
            config.useAsyncTransfers = true;      // Parallel operations
        } else {
            config.subimageSize = 256;            // Optimized for CPU
            // CPU-specific optimizations can be added here
        }
        
        config.gpu = backend;
        algorithm.configure(config);
    }
    
    void configureBackendSpecific(DeconvolutionRequest& request, 
                                const std::string& backend) {
        if (backend == "cuda") {
            request.save_subimages = false;  // Reduce I/O for GPU processing
            
            // We can add GPU-specific settings here if needed
            auto& config = *request.setupConfig->deconvolutionConfig;
            config.useCUBEKernels = true;
            config.enableErrorChecking = false;  // Disable for max performance
        }
        
        request.save_separate = false;
        request.output_path = "result.tif";
    }
};

int main() {
    try {
        // Demonstrate different backend selection strategies
        
        std::string imagePath = "large_image.tif";
        std::string psfPath = "psf.tif";
        
        // 1. Auto mode (smart selection)
        std::cout << "=== Auto Mode ===" << std::endl;
        DeconvolutionProcessor autoProcessor(DeconvolutionProcessor::AUTO);
        std::cout << "GPU Available: " << (autoProcessor.isGPUAvailable() ? "Yes" : "No") << std::endl;
        
        DeconvolutionResult autoResult = autoProcessor.processImage(imagePath, psfPath);
        std::cout << "Auto processing: " << (autoResult.success ? "Success" : "Failed") << std::endl;
        
        // 2. Force GPU mode
        if (autoProcessor.isGPUAvailable()) {
            std::cout << "\n=== GPU Mode ===" << std::endl;
            DeconvolutionProcessor gpuProcessor(DeconvolutionProcessor::GPU);
            DeconvolutionResult gpuResult = gpuProcessor.processImage(imagePath, psfPath, "rltv", 200);
            std::cout << "GPU processing: " << (gpuResult.success ? "Success" : "Failed") << std::endl;
        }
        
        // 3. Force CPU mode
        std::cout << "\n=== CPU Mode ===" << std::endl;
        DeconvolutionProcessor cpuProcessor(DeconvolutionProcessor::CPU);
        DeconvolutionResult cpuResult = cpuProcessor.processImage(imagePath, psfPath, "rl", 50);
        std::cout << "CPU processing: " << (cpuResult.success ? "Success" : "Failed") << std::endl;
        
        // Performance comparison
        if (autoResult.success && cpuResult.success && autoProcessor.isGPUAvailable()) {
            std::cout << "\n=== Performance Comparison ===" << std::endl;
            std::cout << "Speedup (GPU vs CPU): " 
                      << (static_cast<double>(cpuResult.processingTime) / autoResult.processingTime) << "x" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Advanced Configuration Examples

### Multi-PSF Configuration

```cpp
#include "Dolphin.h"
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    try {
        Dolphin dolphin;
        
        // Create complex multi-PSF configuration
        std::string imagePath = "multi_layer_image.tif";
        
        // Configure global PSF for all layers
        SetupConfig setupConfig;
        setupConfig.imagePath = imagePath;
        setupConfig.psfFilePath = "global_psf.tif";
        
        // Add layer-specific PSFs using JSON configuration
        setupConfig.psfConfigPath = "multi_psf_config.json";
        
        // Create multi-PSF configuration JSON
        json psfConfig = {
            {
                "default_psf", {
                    {"path", "global_psf.tif"},
                    {"psfx", 51},
                    {"psfy", 51},
                    {"psfz", 51},
                    {"psfmodel", "gaussian"}
                }
            },
            {
                "layer_psfs", {
                    {
                        "psf_1", {
                            {"path", "layer1_psf.tif"},
                            {"layers", {2, 3, 4}},
                            {"subimages", {1, 2, 10, 11}}
                        }
                    },
                    {
                        "psf_2", {
                            {"path", "layer2_psf.tif"},
                            {"layers", {5, 6, 7}},
                            {"subimages", {3, 4, 5}}
                        }
                    }
                }
            }
        };
        
        // Write multi-PSF configuration
        std::ofstream configFile("multi_psf_config.json");
        configFile << psfConfig.dump(4);
        configFile.close();
        
        // Configure algorithm with multi-PSF support
        auto algorithmFactory = DeconvolutionAlgorithmFactory::getInstance();
        auto algorithm = algorithmFactory.create("rltv", "none");
        
        DeconvolutionConfig config;
        config.algorithmName = "rltv";
        config.iterations = 75;
        config.lambda = 0.015;
        config.grid = true;
        config.subimageSize = 256;
        config.time = true;
        config.saveSubimages = false;
        
        algorithm->configure(config);
        
        // Set up deconvolution request
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>(setupConfig);
        request.setupConfig->psfConfigPath = "multi_psf_config.json";
        request.save_separate = true;  // Save each layer separately
        
        // Run multi-PSF deconvolution
        auto start = std::chrono::high_resolution_clock::now();
        DeconvolutionResult result = dolphin.deconvolve(request);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (result.success) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Multi-PSF deconvolution completed in " << duration.count() << " ms" << std::endl;
            
            // Save results
            result.hyperstack.saveAsTifFile("multi_psf_result.tif");
            
            // Save individual layers if requested
            if (request.save_separate) {
                for (size_t i = 0; i < result.hyperstack.channels.size(); ++i) {
                    std::string layerFilename = "layer_" + std::to_string(i) + ".tif";
                    result.hyperstack.channels[i].image.saveAsTifFile(layerFilename);
                }
            }
        } else {
            std::cerr << "Multi-PSF processing failed: " << result.errorMessage << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### High-Performance Optimization Configuration

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#include <iostream>
#include <fstream>

class PerformanceOptimizationHelper {
public:
    static json createOptimalCPUConfig() {
        json config;
        
        // Basic settings
        config["algorithm"] = "rltv";
        config["iterations"] = 100;
        config["lambda"] = 0.01;
        config["epsilon"] = 1e-6;
        
        // Time and monitoring
        config["time"] = true;
        
        // Grid processing for large datasets
        config["grid"] = true;
        config["subimageSize"] = 256;  // Optimized for CPU cache
        config["borderType"] = 2;      // Reflecting borders
        
        // CPU-specific optimizations
        config["cpu_optimizations"] = {
            {"optimizePlans", true},     // Use FFTW_MEASURE
            {"ompThreads", -1},          // Use all available cores
            {"enableMonitoring", true},
            {"memoryEfficientMode", false}
        };
        
        return config;
    }
    
    static json createOptimalGPUConfig() {
        json config;
        
        // Basic settings
        config["algorithm"] = "rltv";
        config["iterations"] = 200;
        config["lambda"] = 0.015;
        config["epsilon"] = 1e-6;
        
        // Time and monitoring
        config["time"] = true;
        
        // Grid processing for GPU parallelism
        config["grid"] = true;
        config["subimageSize"] = 512;  // Larger for GPU parallelism
        config["borderType"] = 2;
        
        // GPU-specific optimizations
        config["gpu"] = "cuda";
        config["gpu_optimizations"] = {
            {"usePinnedMemory", true},
            {"useAsyncTransfers", true},
            {"useCUBEKernels", true},
            {"optimizePlans", true},
            {"enableErrorChecking", false},  // Disable for max performance
            {"preferredGPUDevice", 0},
            {"maxGPUStreams", 8},
            {"enablePerformanceTracking", true}
        };
        
        return config;
    }
    
    static json createBalancedConfig() {
        json config;
        
        // Adaptive settings
        config["algorithm"] = "rl";
        config["iterations"] = 75;
        config["epsilon"] = 1e-6;
        config["time"] = true;
        
        // Smart grid processing
        config["grid"] = "auto";  // Could be interpreted by processor
        config["subimageSize"] = 0;  // Auto-adjust to PSF size
        config["borderType"] = 2;
        
        // Auto backend selection
        config["backend"] = "auto";
        
        return config;
    }
    
    static bool applyPerformanceOptimization(const std::string& configPath) {
        try {
            // Load existing configuration
            std::ifstream configFile(configPath);
            if (!configFile.is_open()) {
                std::cerr << "Cannot open config file: " << configPath << std::endl;
                return false;
            }
            
            json config;
            configFile >> config;
            configFile.close();
            
            // Apply auto-optimization based on system capabilities
            auto factory = DeconvolutionAlgorithmFactory::getInstance();
            
            if (factory.isGPUSupported()) {
                std::cout << "GPU detected, applying GPU optimizations..." << std::endl;
                
                // Merge with optimal GPU config
                json optimalGPU = createOptimalGPUConfig();
                mergeConfig(config, optimalGPU);
                
                // Detect optimal GPU device
                auto availableGPUs = factory.getGPUAlgorithms();
                if (availableGPUs.size() > 0) {
                    config["gpu_optimizations"]["preferredGPUDevice"] = 0;
                }
                
                config["gpu"] = "cuda";
            } else {
                std::cout << "CPU detected, applying CPU optimizations..." << std::endl;
                
                // Merge with optimal CPU config
                json optimalCPU = createOptimalCPUConfig();
                mergeConfig(config, optimalCPU);
                
                config["gpu"] = "none";
            }
            
            // Save optimized configuration
            std::ofstream optimizedConfig(configPath + ".optimized");
            optimizedConfig << config.dump(4);
            optimizedConfig.close();
            
            std::cout << "Optimization applied and saved to " << configPath << ".optimized" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Performance optimization failed: " << e.what() << std::endl;
            return false;
        }
    }
    
private:
    static void mergeConfig(json& target, const json& source) {
        // Recursive merge function
        for (auto& [key, value] : source.items()) {
            if (source.contains(key) && !source[key].is_null()) {
                if (target.contains(key) && target[key].is_object() && source[key].is_object()) {
                    mergeConfig(target[key], source[key]);
                } else {
                    target[key] = value;
                }
            }
        }
    }
};

int main() {
    try {
        std::string configPath = "user_config.json";
        
        // Create base configuration
        json baseConfig = {
            {"algorithm", "rltv"},
            {"iterations", 50},
            {"grid", true},
            {"time", true}
        };
        
        // Write base configuration
        std::ofstream configFile(configPath);
        configFile << baseConfig.dump(4);
        configFile.close();
        
        std::cout << "Created base configuration: " << configPath << std::endl;
        
        // Apply performance optimizations
        if (PerformanceOptimizationHelper::applyPerformanceOptimization(configPath)) {
            std::cout << "Performance optimization completed successfully" << std::endl;
        }
        
        // Load and use optimized configuration
        auto dolphin = std::make_unique<Dolphin>();
        
        // Load optimized configuration
        auto setupConfig = SetupConfig::createFromJSONFile(configPath + ".optimized");
        
        // Set up deconvolution request
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>(*setupConfig);
        request.save_separate = false;
        
        // Run deconvolution with optimized settings
        std::cout << "Running deconvolution with optimized configuration..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        DeconvolutionResult result = dolphin->deconvolve(request);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (result.success) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Optimized processing completed in " << duration.count() << " ms" << std::endl;
            
            // Save performance report
            json performanceReport = {
                {"config_file", configPath + ".optimized"},
                {"backend", setupConfig->gpu},
                {"processing_time_ms", duration.count()},
                {"success", true},
                {"result_file", "optimized_result.tif"}
            };
            
            std::ofstream reportFile("performance_report.json");
            reportFile << performanceReport.dump(4);
            reportFile.close();
            
            std::cout << "Performance report saved to performance_report.json" << std::endl;
        } else {
            std::cerr << "Optimized processing failed: " << result.errorMessage << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Performance Monitoring Examples

### CPU Performance Monitoring

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h"
#include <iostream>
#include <chrono>
#include <iomanip>

class PerformanceMonitor {
public:
    void startCPUBenchmark(const std::string& algorithmName, int iterations, int subimageSize = 256) {
        std::cout << "=== CPU Performance Benchmark ===" << std::endl;
        std::cout << "Algorithm: " << algorithmName << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Subimage Size: " << subimageSize << "x" << subimageSize << std::endl;
        
        // Create algorithm and configure for CPU
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        auto algorithm = factory.create(algorithmName, "none");
        
        if (!algorithm) {
            std::cerr << "Failed to create algorithm" << std::endl;
            return;
        }
        
        // Configure with CPU optimizations
        DeconvolutionConfig config;
        config.algorithmName = algorithmName;
        config.iterations = iterations;
        config.grid = true;
        config.subimageSize = subimageSize;
        config.time = true;
        config.optimizePlans = true;
        
        algorithm->configure(config);
        
        // Cast to CPU backend for detailed monitoring
        auto* cpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmCPU*>(algorithm.get());
        if (cpuAlgorithm) {
            std::cout << "CPU-specific optimizations enabled" << std::endl;
            cpuAlgorithm->logMemoryUsage("pre_processing");
        }
        
        // Create test data
        auto testData = createTestData();
        
        // Run benchmark
        auto benchmarkStart = std::chrono::high_resolution_clock::now();
        
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = testData.imagePath;
        request.setupConfig->psfFilePath = testData.psfPath;
        request.setupConfig->gpu = "none";
        
        DeconvolutionResult result = Dolphin().deconvolve(request);
        
        auto benchmarkEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(benchmarkEnd - benchmarkStart);
        
        // Analyze and report performance
        PerformanceAnalysis analysis;
       .analysis.totalTime = totalDuration.count();
        analysis.algorithmName = algorithmName;
        analysis.iterations = iterations;
        analysis.subimageSize = subimageSize;
        analysis.success = result.success;
        
        if (cpuAlgorithm) {
            analysis.memoryUsage = "Available from CPU backend monitoring";
            // Get additional metrics from CPU backend
        }
        
        printPerformanceReport(analysis);
        
        // Store for comparison
        m_cpuBenchmarks.push_back(analysis);
        
        // Cleanup
        testData.cleanup();
    }
    
    void compareCPUBenchmarks() {
        if (m_cpuBenchmarks.empty()) {
            std::cout << "No CPU benchmarks available for comparison" << std::endl;
            return;
        }
        
        std::cout << "\n=== CPU Benchmark Comparison ===" << std::endl;
        
        // Find best and worst performance
        auto minTime = std::min_element(m_cpuBenchmarks.begin(), m_cpuBenchmarks.end(),
                                    [](const auto& a, const auto& b) {
                                        return a.totalTime < b.totalTime;
                                    });
        
        auto maxTime = std::max_element(m_cpuBenchmarks.begin(), m_cpuBenchmarks.end(),
                                    [](const auto& a, const auto& b) {
                                        return a.totalTime < b.totalTime;
                                    });
        
        std::cout << "Fastest Configuration:" << std::endl;
        printBenchmarkSummary(*minTime);
        
        std::cout << "\nSlowest Configuration:" << std::endl;
        printBenchmarkSummary(*maxTime);
        
        // Calculate average performance
        double avgTime = 0.0;
        for (const auto& benchmark : m_cpuBenchmarks) {
            avgTime += benchmark.totalTime;
        }
        avgTime /= m_cpuBenchmarks.size();
        
        std::cout << "\nAverage Processing Time: " << std::fixed << std::setprecision(2) 
                  << avgTime << " ms" << std::endl;
    }
    
private:
    struct PerformanceAnalysis {
        long totalTime;
        std::string algorithmName;
        int iterations;
        int subimageSize;
        bool success;
        std::string memoryUsage;
    };
    
    std::vector<PerformanceAnalysis> m_cpuBenchmarks;
    
    struct TestData {
        std::string imagePath;
        std::string psfPath;
        
        TestData(const std::string& imgPath, const std::string& psfPath)
            : imagePath(imgPath), psfPath(psfPath) {}
        
        void cleanup() {
            // Cleanup test data files if they were created
        }
    };
    
    TestData createTestData() {
        // In a real implementation, this would create or load test data
        return TestData("benchmark_input.tif", "benchmark_psf.tif");
    }
    
    void printPerformanceReport(const PerformanceAnalysis& analysis) {
        std::cout << "\nPerformance Report:" << std::endl;
        std::cout << "==================" << std::endl;
        std::cout << "Algorithm: " << analysis.algorithmName << std::endl;
        std::cout << "Total Time: " << analysis.totalTime << " ms" << std::endl;
        std::cout << "Status: " << (analysis.success ? "Success" : "Failed") << std::endl;
        std::cout << "Time per Iteration: " << (analysis.totalTime / analysis.iterations) << " ms" << std::endl;
        std::cout << "Memory Usage: " << analysis.memoryUsage << std::endl;
        
        // Calculate performance metrics
        double throughput = 1.0 / (analysis.totalTime / 1000.0);  // Images per second
        std::cout << "Throughput: " << std::fixed << std::setprecision(4) << throughput << " images/sec" << std::endl;
    }
    
    void printBenchmarkSummary(const PerformanceAnalysis& benchmark) {
        std::cout << "  Algorithm: " << benchmark.algorithmName << std::endl;
        std::cout << "  Time: " << benchmark.totalTime << " ms" << std::endl;
        std::cout << "  Iterations: " << benchmark.iterations << std::endl;
        std::cout << "  Subimage Size: " << benchmark.subimageSize << std::endl;
        std::cout << "  Status: " << (benchmark.success ? "Success" : "Failed") << std::endl;
    }
};

int main() {
    PerformanceMonitor monitor;
    
    try {
        // Run different CPU configurations for benchmarking
        monitor.startCPUBenchmark("rl", 50, 128);   // Small subimages
        monitor.startCPUBenchmark("rl", 100, 256);  // Medium subimages
        monitor.startCPUBenchmark("rl", 50, 512);   // Large subimages
        monitor.startCPUBenchmark("rltv", 75, 256); // Different algorithm
        
        // Compare results
        monitor.compareCPUBenchmarks();
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### GPU Performance Monitoring

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#include <iostream>
#include <iomanip>
#include <fstream>

class GPUPerformanceMonitor {
public:
    bool initialize() {
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        
        if (!factory.isGPUSupported()) {
            std::cout << "GPU not available, falling back to CPU monitoring" << std::endl;
            return false;
        }
        
        // Initialize GPU device information
        m_gpuAvailable = true;
        initializeDevices();
        
        std::cout << "GPU Performance Monitor Initialized" << std::endl;
        listGPUDevices();
        
        return true;
    }
    
    void runGPUBenchmark(const std::string& algorithmName, const std::string& gpuConfig = "default") {
        if (!m_gpuAvailable) {
            std::cout << "GPU not available, skipping GPU benchmark" << std::endl;
            return;
        }
        
        std::cout << "\n=== GPU Performance Benchmark ===" << std::endl;
        std::cout << "Algorithm: " << algorithmName << std::endl;
        std::cout << "Configuration: " << gpuConfig << std::endl;
        
        // Select optimal GPU for benchmarking
        int selectedDevice = selectBenchmarkDevice();
        
        // Create GPU algorithm
        auto algorithm = createGPUAlgorithm(algorithmName, selectedDevice);
        if (!algorithm) {
            std::cerr << "Failed to create GPU algorithm" << std::endl;
            return;
        }
        
        // Configure for benchmarking
        auto request configureGPURequest(algorithm.get(), gpuConfig);
        
        // Get GPU backend for detailed monitoring
        auto* gpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmGPU*>(algorithm.get());
        if (gpuAlgorithm) {
            // Monitor memory usage before processing
            gpuAlgorithm->logMemoryUsage("pre_benchmark");
            std::cout << "GPU memory usage logged before benchmark" << std::endl;
        }
        
        // Run benchmark with detailed timing
        auto benchmarkStart = std::chrono::high_resolution_clock::now();
        
        DeconvolutionResult result = Dolphin().deconvolve(request);
        
        auto benchmarkEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(benchmarkEnd - benchmarkEnd);
        
        // Analyze GPU performance
        GPUPerformanceAnalysis analysis;
        analysis.algorithmName = algorithmName;
        analysis.config = gpuConfig;
        analysis.totalTime = totalDuration.count();
        analysis.deviceId = selectedDevice;
        analysis.success = result.success;
        
        // Get GPU-specific metrics
        if (gpuAlgorithm) {
            analysis.lastGPURuntime = gpuAlgorithm->getLastGPURuntime();
            analysis.runtimes = gpuAlgorithm->getGPURuntimeHistory();
            analysis.peakMemory = gpuAlgorithm->getPeakGPUMemory();
            
            // Log final GPU metrics
            gpuAlgorithm->logPerformanceMetrics();
            gpuAlgorithm->logMemoryUsage("post_benchmark");
        }
        
        // Generate and save comprehensive report
        printGPUPerformanceReport(analysis);
        saveDetailedReport(analysis, request);
        
        // Store for comparison
        m_gpuBenchmarks.push_back(analysis);
    }
    
    void generateComprehensiveReport() {
        if (m_gpuBenchmarks.empty()) {
            std::cout << "No GPU benchmarks available for comprehensive analysis" << std::endl;
            return;
        }
        
        std::cout << "\n=== Comprehensive GPU Performance Report ===" << std::endl;
        
        // Overall statistics
        computeOverallStatistics();
        
        // Device-specific analysis
        analyzeDevicePerformance();
        
        // Algorithm comparison
        compareAlgorithms();
        
        // Config optimization recommendations
        generateOptimizationRecommendations();
        
        // Save comprehensive report
        saveComprehensiveReport();
    }
    
private:
    struct GPUPerformanceAnalysis {
        std::string algorithmName;
        std::string config;
        long totalTime;
        int deviceId;
        bool success;
        double lastGPURuntime;
        std::vector<double> runtimes;
        size_t peakMemory;
        
        // Performance metrics
        double avgMemoryBandwidth() const {
            if (runtimes.empty()) return 0.0;
            // Calculate effective memory bandwidth based on transfers and timing
            return 0.0; // Implementation would calculate actual bandwidth
        }
        
        double effectiveGPUUtilization() const {
            if (runtimes.empty()) return 0.0;
            // Calculate GPU utilization based on theoretical vs actual processing time
            return 0.0; // Implementation would calculate actual utilization
        }
    };
    
    bool m_gpuAvailable = false;
    std::vector<int> m_availableDevices;
    std::vector<GPUPerformanceAnalysis> m_gpuBenchmarks;
    
    void initializeDevices() {
        m_availableDevices.clear();
        for (int i = 0; i < 10; i++) {  // Reasonable limit
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, i);
            if (err == cudaSuccess) {
                m_availableDevices.push_back(i);
            }
        }
    }
    
    void listGPUDevices() const {
        std::cout << "Available GPU Devices:" << std::endl;
        std::cout << "======================" << std::endl;
        
        for (int device : m_availableDevices) {
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, device);
            if (err == cudaSuccess) {
                std::cout << "Device " << device << ": " << props.name << std::endl;
                std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
                std::cout << "  Memory: " << (props.totalGlobalMem / 1024.0 / 1024.0) << " MB" << std::endl;
                std::cout << "  Multiprocessors: " << props.multiProcessorCount << std::endl;
            }
        }
    }
    
    int selectBenchmarkDevice() const {
        if (m_availableDevices.empty()) return -1;
        
        // Select device with most memory and highest compute capability
        int bestDevice = m_availableDevices[0];
        size_t bestScore = 0;
        
        for (int device : m_availableDevices) {
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, device);
            
            if (err == cudaSuccess) {
                size_t score = props.totalGlobalMem + 
                             (props.major * 10 + props.minor) * 1000000ULL +
                             props.multiProcessorCount * 10000ULL;
                
                if (score > bestScore) {
                    bestScore = score;
                    bestDevice = device;
                }
            }
        }
        
        return bestDevice;
    }
    
    std::unique_ptr<BaseDeconvolutionAlgorithm> createGPUAlgorithm(const std::string& algorithmName, int deviceId) {
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        auto algorithm = factory.create(algorithmName, "cuda");
        
        if (algorithm) {
            // Configure GPU-specific settings
            DeconvolutionConfig config;
            config.algorithmName = algorithmName;
            config.gpu = "cuda";
            config.time = true;
            config.grid = true;
            
            auto* gpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmGPU*>(algorithm.get());
            if (gpuAlgorithm) {
                gpuAlgorithm->setGPUDevice(deviceId);
                gpuAlgorithm->resetGPUStats();
            }
            
            algorithm->configure(config);
        }
        
        return algorithm;
    }
    
    DeconvolutionRequest configureGPURequest(BaseDeconvolutionAlgorithm* algorithm, const std::string& config) {
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = "gpu_benchmark_input.tif";
        request.setupConfig->psfFilePath = "gpu_benchmark_psf.tif";
        request.setupConfig->gpu = "cuda";
        
        // Apply configuration-specific settings
        if (config == "high_performance") {
            request.setupConfig->deconvolutionConfig = std::make_shared<DeconvolutionConfig>();
            request.setupConfig->deconvolutionConfig->usePinnedMemory = true;
            request.setupConfig->deconvolutionConfig->useAsyncTransfers = true;
            request.setupConfig->deconvolutionConfig->useCUBEKernels = true;
            request.setupConfig->deconvolutionConfig->enableErrorChecking = false;
        } else if (config == "balanced") {
            // Balanced configuration settings
        }
        
        return request;
    }
    
    void printGPUPerformanceReport(const GPUPerformanceAnalysis& analysis) {
        std::cout << "\nGPU Performance Report:" << std::endl;
        std::cout << "=======================" << std::endl;
        std::cout << "Algorithm: " << analysis.algorithmName << std::endl;
        std::cout << "Configuration: " << analysis.config << std::endl;
        std::cout << "Device ID: " << analysis.deviceId << std::endl;
        std::cout << "Total Time: " << analysis.totalTime << " ms" << std::endl;
        std::cout << "Last GPU Runtime: " << std::fixed << std::setprecision(3) 
                  << analysis.lastGPURuntime << " ms" << std::endl;
        std::cout << "Peak Memory Usage: " << (analysis.peakMemory / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Status: " << (analysis.success ? "Success" : "Failed") << std::endl;
        
        if (!analysis.runtimes.empty()) {
            double avgRuntime = std::accumulate(analysis.runtimes.begin(), analysis.runtimes.end(), 0.0) / analysis.runtimes.size();
            std::cout << "Average GPU Runtime: " << std::fixed << std::setprecision(3) << avgRuntime << " ms" << std::endl;
            std::cout << "GPU Operations Count: " << analysis.runtimes.size() << std::endl;
        }
    }
    
    void saveDetailedReport(const GPUPerformanceAnalysis& analysis, const DeconvolutionRequest& request) {
        std::string filename = "gpu_benchmark_" + analysis.algorithmName + "_report.json";
        
        json report = {
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()},
            {"algorithm", analysis.algorithmName},
            {"configuration", analysis.config},
            {"device_id", analysis.deviceId},
            {"total_time_ms", analysis.totalTime},
            {"success", analysis.success},
            {"gpu_metrics", {
                {"last_runtime_ms", analysis.lastGPURuntime},
                {"peak_memory_bytes", analysis.peakMemory},
                {"operation_count", analysis.runtimes.size()}
            }},
            {"request_config", {
                {"image_path", request.setupConfig->imagePath},
                {"auto_backend_selection", request.setupConfig->gpu}
            }}
        };
        
        if (!analysis.runtimes.empty()) {
            double avgRuntime = std::accumulate(analysis.runtimes.begin(), analysis.runtimes.end(), 0.0) / analysis.runtimes.size();
            double minRuntime = *std::min_element(analysis.runtimes.begin(), analysis.runtimes.end());
            double maxRuntime = *std::max_element(analysis.runtimes.begin(), analysis.runtimes.end());
            
            report["gpu_metrics"]["average_runtime_ms"] = avgRuntime;
            report["gpu_metrics"]["min_runtime_ms"] = minRuntime;
            report["gpu_metrics"]["max_runtime_ms"] = maxRuntime;
        }
        
        std::ofstream reportFile(filename);
        reportFile << report.dump(4);
        reportFile.close();
        
        std::cout << "Detailed report saved to: " << filename << std::endl;
    }
    
    void computeOverallStatistics() {
        std::cout << "\nOverall GPU Performance Statistics:" << std::endl;
        
        double totalProcessingTime = 0.0;
        size_t totalMemoryUsage = 0;
        size_t successfulRuns = 0;
        
        for (const auto& benchmark : m_gpuBenchmarks) {
            totalProcessingTime += benchmark.totalTime;
            totalMemoryUsage += benchmark.peakMemory;
            if (benchmark.success) {
                successfulRuns++;
            }
        }
        
        size_t numBenchmarks = m_gpuBenchmarks.size();
        double avgProcessingTime = totalProcessingTime / numBenchmarks;
        double avgMemoryUsage = totalMemoryUsage / numBenchmarks;
        double successRate = (static_cast<double>(successfulRuns) / numBenchmarks) * 100.0;
        
        std::cout << "Total Benchmarks: " << numBenchmarks << std::endl;
        std::cout << "Average Processing Time: " << std::fixed << std::setprecision(3) << avgProcessingTime << " ms" << std::endl;
        std::cout << "Average Memory Usage: " << (avgMemoryUsage / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1) << successRate << "%" << std::endl;
    }
    
    void analyzeDevicePerformance() {
        std::map<int, std::vector<GPUPerformanceAnalysis>> deviceBenchmarks;
        
        // Group benchmarks by device
        for (const auto& benchmark : m_gpuBenchmarks) {
            deviceBenchmarks[benchmark.deviceId].push_back(benchmark);
        }
        
        std::cout << "\nDevice-Specific Performance Analysis:" << std::endl;
        
        for (auto& [deviceId, benchmarks] : deviceBenchmarks) {
            if (benchmarks.empty()) continue;
            
            double avgTime = 0.0;
            size_t avgMemory = 0;
            size_t successfulCount = 0;
            
            for (const auto& benchmark : benchmarks) {
                avgTime += benchmark.totalTime;
                avgMemory += benchmark.peakMemory;
                if (benchmark.success) {
                    successfulCount++;
                }
            }
            
            avgTime /= benchmarks.size();
            avgMemory /= benchmarks.size();
            double successRate = (static_cast<double>(successfulCount) / benchmarks.size()) * 100.0;
            
            // Get device name
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
            std::string deviceName = (err == cudaSuccess) ? props.name : "Unknown";
            
            std::cout << "Device " << deviceId << " (" << deviceName << "):" << std::endl;
            std::cout << "  Average Time: " << avgTime << " ms" << std::endl;
            std::cout << "  Average Memory: " << (avgMemory / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "  Success Rate: " << successRate << "%" << std::endl;
            std::cout << "  Operations: " << benchmarks.size() << std::endl;
        }
    }
    
    void compareAlgorithms() {
        std::map<std::string, std::vector<GPUPerformanceAnalysis>> algorithmBenchmarks;
        
        // Group benchmarks by algorithm
        for (const auto& benchmark : m_gpuBenchmarks) {
            algorithmBenchmarks[benchmark.algorithmName].push_back(benchmark);
        }
        
        std::cout << "\nAlgorithm Comparison GPU Performance:" << std::endl;
        
        for (auto& [algorithm, benchmarks] : algorithmBenchmarks) {
            if (benchmarks.empty()) continue;
            
            double avgTime = 0.0;
            size_t successCount = 0;
            
            for (const auto& benchmark : benchmarks) {
                avgTime += benchmark.totalTime;
                if (benchmark.success) {
                    successCount++;
                }
            }
            
            avgTime /= benchmarks.size();
            double successRate = (static_cast<double>(successCount) / benchmarks.size()) * 100.0;
            
            std::cout << algorithm << ":" << std::endl;
            std::cout << "  Average GPU Processing: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
            std::cout << "  Success Rate: " << successRate << "%" << std::endl;
            std::cout << "  Benchmarks: " << benchmarks.size() << std::endl;
        }
    }
    
    void generateOptimizationRecommendations() {
        std::cout << "\nGPU Optimization Recommendations:" << std::endl;
        
        // Analyze patterns in benchmark data to provide recommendations
        if (m_gpuBenchmarks.size() < 3) {
            std::cout << "Insufficient data for recommendations - run more benchmarks" << std::endl;
            return;
        }
        
        // Analyze across different configurations
        std::map<std::string, std::vector<double>> avgRuntimesByConfig;
        
        for (const auto& benchmark : m_gpuBenchmarks) {
            avgRuntimesByConfig[benchmark.config].push_back(benchmark.lastGPURuntime);
        }
        
        // Find best performing configuration
        std::string bestConfig;
        double bestTime = std::numeric_limits<double>::max();
        
        for (auto& [config, times] : avgRuntimesByConfig) {
            if (times.empty()) continue;
            
            double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            
            if (avgTime < bestTime) {
                bestTime = avgTime;
                bestConfig = config;
            }
            
            std::cout << config << " avg runtime: " << avgTime << " ms" << std::endl;
        }
        
        std::cout << "\nRecommendation: Use '" << bestConfig << "' configuration for best performance" << std::endl;
        
        // Generate configuration-specific recommendations
        generateConfigSpecificRecommendations();
    }
    
    void generateConfigSpecificRecommendations() {
        std::cout << "\nConfiguration-Specific Recommendations:" << std::endl;
        
        // Analyze memory vs performance trade-offs
        size_t lowMemoryThreshold = 1024ULL * 1024 * 1024;  // 1GB
        size_t highMemoryThreshold = 1024ULL * 1024 * 1024 * 8;  // 8GB
        
        for (const auto& benchmark : m_gpuBenchmarks) {
            if (!benchmark.success) continue;
            
            // Memory usage categorization
            if (benchmark.peakMemory < lowMemoryThreshold) {
                // Low memory usage, potentially can scale up for better performance
                std::cout << "Low memory usage (" << (benchmark.peakMemory / (1024.0 * 1024.0)) 
                          << " MB) - Consider larger subimages or more iterations" << std::endl;
            } else if (benchmark.peakMemory > highMemoryThreshold) {
                // High memory usage - recommend optimization
                std::cout << "High memory usage (" << (benchmark.peakMemory / (1024.0 * 1024.0)) 
                          << " MB) - Consider smaller subimages or async transfers" << std::endl;
            }
        }
    }
    
    void saveComprehensiveReport() {
        std::string filename = "gpu_performance_comprehensive_report.json";
        
        json comprehensiveReport = {
            {"report_type", "gpu_performance_comprehensive"},
            {"generated_at", std::chrono::system_clock::now().time_since_epoch().count()},
            {"gpu_enabled", m_gpuAvailable},
            {"total_benchmarks", m_gpuBenchmarks.size()},
            {"individual_benchmarks", json::array()}
        };
        
        // Add individual benchmark data
        for (const auto& benchmark : m_gpuBenchmarks) {
            json benchmarkEntry = {
                {"algorithm_name", benchmark.algorithmName},
                {"configuration", benchmark.config},
                {"device_id", benchmark.deviceId},
                {"total_time_ms", benchmark.totalTime},
                {"last_gpu_runtime_ms", benchmark.lastGPURuntime},
                {"peak_memory_bytes", benchmark.peakMemory},
                {"operation_count", benchmark.runtimes.size()},
                {"success", benchmark.success}
            };
            
            if (!benchmark.runtimes.empty()) {
                benchmarkEntry["runtimes"] = benchmark.runtimes;
                benchmarkEntry["min_runtime_ms"] = *std::min_element(benchmark.runtimes.begin(), benchmark.runtimes.end());
                benchmarkEntry["max_runtime_ms"] = *std::max_element(benchmark.runtimes.begin(), benchmark.runtimes.end());
            }
            
            comprehensiveReport["individual_benchmarks"].push_back(benchmarkEntry);
        }
        
        // Add summary statistics
        computeAndAddSummaryStats(comprehensiveReport);
        
        std::ofstream reportFile(filename);
        reportFile << comprehensiveReport.dump(4);
        reportFile.close();
        
        std::cout << "\nComprehensive report saved to: " << filename << std::endl;
    }
    
    void computeAndAddSummaryStats(json& report) {
        if (m_gpuBenchmarks.empty()) return;
        
        double totalTimeSum = 0.0;
        size_t peakMemorySum = 0;
        size_t successCount = 0;
        
        for (const auto& benchmark : m_gpuBenchmarks) {
            totalTimeSum += benchmark.totalTime;
            peakMemorySum += benchmark.peakMemory;
            if (benchmark.success) {
                successCount++;
            }
        }
        
        size_t totalBenchmarks = m_gpuBenchmarks.size();
        
        report["summary_statistics"] = {
            {"total_time_ms", totalTimeSum},
            {"average_time_ms", totalTimeSum / totalBenchmarks},
            {"total_memory_bytes", peakMemorySum},
            {"average_memory_bytes", peakMemorySum / totalBenchmarks},
            {"success_count", static_cast<int>(successCount)},
            {"success_rate_percent", (static_cast<double>(successCount) / totalBenchmarks) * 100.0}
        };
    }
};

int main() {
    GPUPerformanceMonitor monitor;
    
    try {
        // Initialize GPU monitoring
        if (!monitor.initialize()) {
            std::cout << "GPU monitoring not available, exiting" << std::endl;
            return 0;
        }
        
        // Run various GPU benchmarks
        monitor.runGPUBenchmark("rl", "high_performance");
        monitor.runGPUBenchmark("rltv", "high_performance");
        monitor.runGPUBenchmark("rl", "balanced");
        monitor.runGPUBenchmark("rltv", "balanced");
        
        // Generate comprehensive analysis
        monitor.generateComprehensiveReport();
        
    } catch (const std::exception& e) {
        std::cerr << "GPU performance monitoring error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Error Handling Examples

### Comprehensive Error Handling

```cpp
#include "Dolphin.h"
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#include <iostream>
#include <stdexcept>
#include <memory>

class SafeDeconvolutionProcessor {
public:
    enum ProcessingMode { AUTO, CPU_ONLY, GPU_ONLY };
    
    SafeDeconvolutionProcessor(ProcessingMode mode = AUTO) : m_processingMode(mode) {
        initialize();
    }
    
    /**
     * Safely execute deconvolution with comprehensive error handling
     */
    DeconvolutionResult processDeconvolution(const DeconvolutionConfig& config,
                                            const std::string& imagePath,
                                            const std::string& psfPath) {
        
        try {
            // Pre-processing validation
            validateInputs(config, imagePath, psfPath);
            
            // Select appropriate backend
            std::string selectedBackend = selectBackend(config);
            
            // Create and validate algorithm
            auto algorithm = createAlgorithm(config, selectedBackend);
            if (!algorithm) {
                throw std::runtime_error("Failed to create algorithm for backend: " + selectedBackend);
            }
            
            // Set up request with validation
            auto request = setupRequest(config, imagePath, psfPath, selectedBackend);
            
            std::cout << "Processing with " << (selectedBackend == "cuda" ? "GPU" : "CPU") 
                      << " backend in " << (config.grid ? "grid" : "single") << " mode" << std::endl;
            
            // Execute deconvolution with monitoring
            return executeWithMonitoring(algorithm, request);
            
        } catch (const std::exception& e) {
            std::cerr << "Deconvolution error: " << e.what() << std::endl;
            return createErrorResult("Processing failed: " + std::string(e.what()));
        }
    }
    
    /**
     * Execute multiple deconvolution attempts with error recovery
     */
    std::vector<DeconvolutionResult> processWithFallbacks(const DeconvolutionConfig& config,
                                                         const std::string& imagePath,
                                                         const std::string& psfPath) {
        
        std::vector<DeconvolutionResult> results;
        
        // Try different backends in order of preference
        std::vector<std::string> backendsToTry = {"cuda", "none"};  // GPU first, then CPU
        
        if (m_processingMode == CPU_ONLY) {
            backendsToTry = {"none"};
        } else if (m_processingMode == GPU_ONLY) {
            backendsToTry = {"cuda"};
        }
        
        for (const std::string& backend : backendsToTry) {
            try {
                DeconvolutionConfig configCopy = config;
                configCopy.gpu = backend;
                
                std::cout << "Attempting deconvolution with " << backend << " backend..." << std::endl;
                DeconvolutionResult result = processDeconvolution(configCopy, imagePath, psfPath);
                
                if (result.success) {
                    std::cout << "Successful deconvolution with " << backend << " backend!" << std::endl;
                    results.push_back(result);
                    break;  // Stop on first success
                } else {
                    std::cout << "Backend " << backend << " failed: " << result.errorMessage << std::endl;
                    results.push_back(result);
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Backend attempt failed: " << backend << " - " << e.what() << std::endl;
                results.push_back(createErrorResult("Backend " + backend + " failed: " + e.what()));
            }
        }
        
        return results;
    }
    
private:
    ProcessingMode m_processingMode;
    bool m_gpuAvailable;
    std::vector<int> m_availableGPUs;
    
    void initialize() {
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        m_gpuAvailable = factory.isGPUSupported();
        
        if (m_gpuAvailable) {
            // Initialize GPU device list
            for (int i = 0; i < 10; i++) {
                cudaDeviceProp props;
                cudaError_t err = cudaGetDeviceProperties(&props, i);
                if (err == cudaSuccess) {
                    m_availableGPUs.push_back(i);
                }
            }
        }
        
        std::cout << "Initialized SafeProcessor - GPU: " << (m_gpuAvailable ? "Available" : "Unavailable") << std::endl;
        if (m_gpuAvailable) {
            std::cout << "Available GPUs: " << m_availableGPUs.size() << std::endl;
        }
    }
    
    void validateInputs(const DeconvolutionConfig& config,
                       const std::string& imagePath,
                       const std::string& psfPath) {
        
        // Validate files exist
        if (!std::ifstream(imagePath).good()) {
            throw std::runtime_error("Input image file not found: " + imagePath);
        }
        
        if (!std::ifstream(psfPath).good()) {
            throw std::runtime_error("PSF file not found: " + psfPath);
        }
        
        // Validate algorithm name
        if (config.algorithmName.empty()) {
            throw std::runtime_error("Algorithm name not specified");
        }
        
        // Validate parameters
        if (config.iterations <= 0) {
            throw std::runtime_error("Iterations must be positive");
        }
        
        if (config.epsilon <= 0 || config.epsilon >= 1) {
            throw std::runtime_error("Epsilon must be between 0 and 1");
        }
        
        // Validate GPU configuration
        if (config.gpu == "cuda" && !m_gpuAvailable) {
            throw std::runtime_error("GPU requested but not available");
        }
        
        if (config.gpu == "cuda" && !m_availableGPUs.empty()) {
            // Validate subimage size compatibility with GPU memory
            size_t estimatedMemory = estimateMemoryRequirements(config);
            size_t availableMemory = getAvailableGPUMemory();
            
            if (estimatedMemory > availableMemory) {
                std::cout << "Warning: Estimated memory requirements (" << (estimatedMemory / (1024.0 * 1024.0))
                          << " MB) exceed available GPU memory (" << (availableMemory / (1024.0 * 1024.0))
                          << " MB)" << std::endl;
            }
        }
    }
    
    std::string selectBackend(const DeconvolutionConfig& config) {
        if (config.gpu == "cuda" && m_gpuAvailable) {
            return "cuda";
        }
        
        if (m_processingMode == GPU_ONLY && m_gpuAvailable) {
            return "cuda";
        }
        
        if (config.gpu == "none" || !m_gpuAvailable) {
            return "none";
        }
        
        // Auto mode - prefer GPU if available
        if (m_processingMode == AUTO && m_gpuAvailable) {
            return "cuda";
        }
        
        return "none";
    }
    
    std::unique_ptr<BaseDeconvolutionAlgorithm> createAlgorithm(const DeconvolutionConfig& config,
                                                                const std::string& backend) {
        
        auto factory = DeconvolutionAlgorithmFactory::getInstance();
        auto algorithm = factory.create(config.algorithmName, backend);
        
        if (!algorithm) {
            throw std::runtime_error("Failed to create algorithm: " + config.algorithmName);
        }
        
        // Additional validation for GPU algorithms
        if (backend == "cuda") {
            auto* gpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmGPU*>(algorithm.get());
            if (gpuAlgorithm && !gpuAlgorithm->isGPUSupported()) {
                throw std::runtime_error("GPU algorithm does not support current device");
            }
        }
        
        return algorithm;
    }
    
    DeconvolutionRequest setupRequest(const DeconvolutionConfig& config,
                                     const std::string& imagePath,
                                     const std::string& psfPath,
                                     const std::string& backend) {
        
        DeconvolutionRequest request;
        request.setupConfig = std::make_shared<SetupConfig>();
        request.setupConfig->imagePath = imagePath;
        request.setupConfig->psfFilePath = psfPath;
        request.setupConfig->gpu = backend;
        request.setupConfig->time = config.time;
        
        // Configure algorithm-specific parameters
        request.setupConfig->deconvolutionConfig = std::make_shared<DeconvolutionConfig>(config);
        
        // Set up optional processing modes
        request.save_separate = config.saveSubimages;
        request.save_subimages = config.saveSubimages;
        
        // Configure GPU-specific optimizations
        if (backend == "cuda") {
            auto& deconvConfig = request.setupConfig->deconvolutionConfig;
            deconvConfig->usePinnedMemory = true;
            deconvConfig->useAsyncTransfers = true;
            deconvConfig->useCUBEKernels = true;
        }
        
        return request;
    }
    
    DeconvolutionResult executeWithMonitoring(std::unique_ptr<BaseDeconvolutionAlgorithm>& algorithm,
                                             const DeconvolutionRequest& request) {
        
        // Monitor memory usage before processing
        logMemoryUsage("pre_processing");
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Execute deconvolution
        DeconvolutionResult result;
        try {
            result = Dolphin().deconvolve(request);
            
            // Monitor performance
            auto endTime = std::chrono::high_resolution_clock::now();
            result.processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            
            // Post-processing validation
            validateResult(result);
            
            logMemoryUsage("post_processing");
            
        } catch (const std::exception& e) {
            result = handleProcessingError(e);
        }
        
        return result;
    }
    
    void validateResult(const DeconvolutionResult& result) {
        if (!result.success) {
            return;
        }
        
        // Validate output data
        if (result.hyperstack.channels.empty()) {
            throw std::runtime_error("Empty output channels in result");
        }
        
        for (const auto& channel : result.hyperstack.channels) {
            if (channel.image.slices.empty()) {
                throw std::runtime_error("Empty image slices in output");
            }
            
            for (const auto& slice : channel.image.slices) {
                // Basic validation of output image
                if (slice.cols <= 0 || slice.rows <= 0) {
                    throw std::runtime_error("Invalid slice dimensions in output");
                }
                
                // Check for NaN values in output
                for (int row = 0; row < slice.rows; ++row) {
                    for (int col = 0; col < slice.cols; ++col) {
                        float value = slice.at<float>(row, col);
                        if (!std::isfinite(value)) {
                            throw std::runtime_error("Invalid (NaN/Inf) value detected in output");
                        }
                    }
                }
            }
        }
    }
    
    DeconvolutionResult handleProcessingError(const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << std::endl;
        
        return createErrorResult(std::string("Processing error: ") + e.what());
    }
    
    DeconvolutionResult createErrorResult(const std::string& errorMessage) {
        DeconvolutionResult result;
        result.success = false;
        result.errorMessage = errorMessage;
        result.processingTime = 0;
        return result;
    }
    
    size_t estimateMemoryRequirements(const DeconvolutionConfig& config) {
        // Simple estimation based on image size and algorithm parameters
        // In real implementation, this would be more sophisticated
        return 1024ULL * 1024 * 1024;  // 1GB default
    }
    
    size_t getAvailableGPUMemory() {
        if (m_availableGPUs.empty()) {
            return 0;
        }
        
        cudaDeviceProp props;
        cudaError_t err = cudaGetDeviceProperties(&props, m_availableGPUs[0]);
        if (err == cudaSuccess) {
            size_t free, total;
            cudaError_t memErr = cudaMemGetInfo(&free, &total);
            if (memErr == cudaSuccess) {
                return free;
            }
        }
        
        return 0;
    }
    
    void logMemoryUsage(const std::string& phase) {
        try {
            auto factory = DeconvolutionAlgorithmFactory::getInstance();
            
            if (factory.isGPUSupported()) {
                auto* gpuAlgorithm = dynamic_cast<BaseDeconvolutionAlgorithmGPU*>(nullptr); 
                // In real implementation, we'd have access to current algorithm instance
                
                // Simulate GPU memory logging
                std::cout << "[MEMORY " << phase << "] GPU memory monitoring would occur here" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Memory monitoring error: " << e.what() << std::endl;
        }
    }
};

int main() {
    try {
        SafeDeconvolutionProcessor processor(SafeDeconvolutionProcessor::AUTO);
        
        // Configure deconvolution with some potentially problematic settings
        DeconvolutionConfig config;
        config.algorithmName = "rltv";
        config.iterations = 100;
        config.epsilon = 1e-6;
        config.lambda = 0.01;
        config.time = true;
        config.grid = true;
        config.subimageSize = 512;  // Large size that might cause memory issues
        config.gpu = "cuda";        // Request GPU
        
        std::string imagePath = "test_image.tif";
        std::string psfPath = "test_psf.tif";
        
        std::cout << "=== Safe Deconvolution Processing Test ===" << std::endl;
        
        // Process with comprehensive error handling and fallbacks
        auto results = processor.processWithFallbacks(config, imagePath, psfPath);
        
        std::cout << "\nProcessing Results Summary:" << std::endl;
        std::cout << "==========================" << std::endl;
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            const std::string& backend = (i < 2) ? (i == 0 ? "GPU" : "CPU") : "Unknown";
            
            std::cout << backend << " Backend:" << std::endl;
            std::cout << "  Success: " << (result.success ? "Yes" : "No") << std::endl;
            std::cout << "  Time: " << result.processingTime << " ms" << std::endl;
            
            if (!result.success) {
                std::cout << "  Error: " << result.errorMessage << std::endl;
            }
            
            std::cout << std::endl;
        }
        
        // Select the first successful result
        auto successfulResult = std::find_if(results.begin(), results.end(),
                                           [](const DeconvolutionResult& r) { return r.success; });
        
        if (successfulResult != results.end()) {
            std::cout << "Using successful result for final output" << std::endl;
            successfulResult->hyperstack.saveAsTifFile("safe_result.tif");
        } else {
            std::cout << "No successful results found" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Critical error in main: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Migration Guide Examples

### Configuration Migration Examples

```cpp
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ConfigurationMigrator {
public:
    /**
     * Migrate old DOLPHIN configuration to new CPU/GPU architecture format
     */
    bool migrateConfiguration(const std::string& oldConfigPath, const std::string& newConfigPath) {
        try {
            // Read old configuration
            std::ifstream oldFile(oldConfigPath);
            if (!oldFile.is_open()) {
                std::cerr << "Cannot open old configuration file: " << oldConfigPath << std::endl;
                return false;
            }
            
            json oldConfig;
            oldFile >> oldConfig;
            oldFile.close();
            
            std::cout << "Reading old configuration from: " << oldConfigPath << std::endl;
            
            // Detect and validate old configuration format
            if (!isOldConfigurationValid(oldConfig)) {
                std::cerr << "Invalid old configuration format" << std::endl;
                return false;
            }
            
            // Convert to new format
            json newConfig = convertToNewFormat(oldConfig);
            
            // Apply auto-optimization if requested
            if (newConfig.contains("auto_optimize") && newConfig["auto_optimize"].get<bool>()) {
                optimizeConfiguration(newConfig);
            }
            
            // Validate new configuration
            if (!isNewConfigurationValid(newConfig)) {
                std::cerr << "Generated configuration validation failed" << std::endl;
                return false;
            }
            
            // Write new configuration
            std::ofstream newFile(newConfigPath);
            newFile << newConfig.dump(4);
            newFile.close();
            
            std::cout << "Successfully migrated to: " << newConfigPath << std::endl;
            
            // Generate migration report
            generateMigrationReport(oldConfig, newConfig, oldConfigPath, newConfigPath);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Configuration migration error: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * Batch migrate multiple configuration files
     */
    bool batchMigrate(const std::vector<std::string>& oldConfigPaths, const std::string& outputDirectory) {
        try {
            std::cout << "Starting batch configuration migration..." << std::endl;
            std:: cout << "Input configurations: " << oldConfigPaths.size() << std::endl;
            
            std::vector<std::string> successfullyMigrated;
            std::vector<std::string> failedToMigrate;
            
            // Create output directory if it doesn't exist
            createDirectory(outputDirectory);
            
            for (const auto& oldPath : oldConfigPaths) {
                std::cout << "Processing: " << oldPath << std::endl;
                
                // Generate output path
                std::string newPath = generateUniquePath(outputDirectory, oldPath, ".migrated.json");
                
                if (migrateConfiguration(oldPath, newPath)) {
                    successfullyMigrated.push_back(oldPath);
                    std::cout << "  ✓ Success" << std::endl;
                } else {
                    failedToMigrate.push_back(oldPath);
                    std::cout << "  ✗ Failed" << std::endl;
                }
            }
            
            // Generate batch report
            generateBatchMigrationReport(successfullyMigrated, failedToMigrate, outputDirectory);
            
            std::cout << "Batch migration completed!" << std::endl;
            std::cout << "Successfully migrated: " << successfullyMigrated.size() << std::endl;
            std::cout << "Failed to migrate: " << failedToMigrate.size() << std::endl;
            
            return failedToMigrate.empty();
            
        } catch (const std::exception& e) {
            std::cerr << "Batch migration error: " << e.what() << std::endl;
            return false;
        }
    }
    
private:
    bool isOldConfigurationValid(const json& config) {
        // Check if it's likely an old DOLPHIN configuration
        bool hasOldFormat = false;
        
        // Check for old-style parameters
        if (config.contains("algorithm") && config["algorithm"].is_string()) {
            hasOldFormat = true;
        }
        
        if (config.contains("iterations") && config["iterations"].is_number()) {
            hasOldFormat = true;
        }
        
        if (config.contains("grid") && config["grid"].is_boolean()) {
            hasOldFormat = true;
        }
        
        // Also check for new-style existence
        if (!config.is_object()) {
            return false;
        }
        
        return hasOldFormat;
    }
    
    bool isNewConfigurationValid(const json& config) {
        // Validate key fields exist
        std::vector<std::string> requiredKeys = {
            "algorithm", "iterations", "gpu", "time", "grid"
        };
        
        for (const auto& key : requiredKeys) {
            if (!config.contains(key)) {
                std::cerr << "Missing required key: " << key << std::endl;
                return false;
            }
        }
        
        // Validate parameter ranges
        if (config["iterations"].get<int>() <= 0) {
            std::cerr << "Invalid iterations value" << std::endl;
            return false;
        }
        
        if (config["epsilon"].get<double>() <= 0 || config["epsilon"].get<double>() >= 1) {
            std::cerr << "Invalid epsilon value" << std::endl;
            return false;
        }
        
        std::string backend = config["gpu"].get<std::string>();
        if (backend != "none" && backend != "cuda" && backend != "auto") {
            std::cerr << "Invalid GPU backend: " << backend << std::endl;
            return false;
        }
        
        return true;
    }
    
    json convertToNewFormat(const json& oldConfig) {
        json newConfig;
        
        // Basic mapping
        newConfig["algorithm"] = oldConfig.value("algorithm", "rl");
        newConfig["iterations"] = oldConfig.value("iterations", 10);
        newConfig["epsilon"] = oldConfig.value("epsilon", 1e-6);
        newConfig["lambda"] = oldConfig.value("lambda", 1e-2);
        newConfig["time"] = oldConfig.value("time", false);
        newConfig["grid"] = oldConfig.value("grid", false);
        
        // Backend selection - convert old backend preferences to new format
        if (oldConfig.contains("gpu")) {
            if (oldConfig["gpu"].get<bool>()) {
                newConfig["gpu"] = "auto";  // Auto-detect best backend
            } else {
                newConfig["gpu"] = "none";   // Force CPU
            }
        } else {
            newConfig["gpu"] = "auto";  // Auto-detect if not specified
        }
        
        // Grid processing parameters
        newConfig["subimageSize"] = oldConfig.value("subimageSize", 0);  // 0 = auto-adjust
        newConfig["borderType"] = oldConfig.value("borderType", 2);
        newConfig["psfSafetyBorder"] = oldConfig.value("psfSafetyBorder", 10);
        
        // Output options
        newConfig["saveSubimages"] = oldConfig.value("savePSF", false) || oldConfig.value("saveSubimages", false);
        newConfig["seperate"] = oldConfig.value("seperate", false);
        
        // New architecture-specific options
        newConfig["auto_optimize"] = false;  // Can be set to true
        
        // GPU-specific options (default values)
        if (newConfig["gpu"].get<std::string>() != "none") {
            newConfig["gpu_optimizations"] = {
                {"usePinnedMemory", false},
                {"useAsyncTransfers", false},
                {"useCUBEKernels", false},
                {"optimizePlans", true},
                {"enableErrorChecking", true}
            };
        }
        
        // CPU-specific options
        newConfig["cpu_optimizations"] = {
            {"optimizePlans", true},
            {"enableMonitoring", false}
        };
        
        // Preserve any unknown configurations with a warning
        migrateUnknownConfigurations(oldConfig, newConfig);
        
        return newConfig;
    }
    
    void optimizeConfiguration(json& config) {
        std::cout << "Applying auto-optimization..." << std::endl;
        
        // Set backend to auto for optimization
        config["gpu"] = "auto";
        
        // Algorithm-specific optimizations
        std::string algorithm = config.value("algorithm", "rl");
        
        if (algorithm == "rl" || algorithm == "rltv") {
            // Iteration count optimization
            if (config["iterations"].get<int>() < 50) {
                config["iterations"] = 75;  // Better baseline for RL
            }
        }
        
        // Grid size optimization
        if (config["subimageSize"].get<int>() == 0) {
            // Auto-adjust - this is optimal for new architecture
            config["subimageSize"] = 0;
        } else {
            // Adjust existing size to optimal ranges
            int subimageSize = config["subimageSize"].get<int>();
            if (subimageSize < 128) {
                config["subimageSize"] = 128;  // Minimum for efficiency
            }
        }
        
        // Enable advanced optimizations for complex algorithms
        if (algorithm == "rltv") {
            config["lambda"] = config.value("lambda", 0.015);
            config["gpu_optimizations"]["useCUBEKernels"] = true;
        }
    }
    
    void migrateUnknownConfigurations(const json& oldConfig, json& newConfig) {
        // Migrate any configuration parameters that aren't recognized
        // Store them under a "legacy" section
        
        json legacyConfig;
        
        // Known keys in new format
        std::vector<std::string> knownKeys = {
            "algorithm", "iterations", "epsilon", "lambda", "time", "grid",
            "gpu", "subimageSize", "borderType", "psfSafetyBorder", "saveSubimages",
            "seperate", "auto_optimize", "gpu_optimizations", "cpu_optimizations"
        };
        
        for (auto it = oldConfig.begin(); it != oldConfig.end(); ++it) {
            if (std::find(knownKeys.begin(), knownKeys.end(), it.key()) == knownKeys.end()) {
                legacyConfig[it.key()] = it.value();
            }
        }
        
        if (!legacyConfig.empty()) {
            newConfig["legacy_parameters"] = legacyConfig;
            std::cout << "Warning: Migrated " << legacyConfig.size() 
                      << " legacy/unrecognized parameters under 'legacy_parameters'" << std::endl;
        }
    }
    
    void generateMigrationReport(const json& oldConfig, const json& newConfig,
                                const std::string& oldPath, const std::string& newPath) {
        
        json report = {
            {"migration_report", true},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()},
            {"source", oldPath},
            {"destination", newPath},
            {"changes_reported", {
                {"algorithm", oldConfig.value("algorithm", "unknown")},
                {"old_backend", oldConfig.value("gpu", "unknown")},
                {"new_backend", newConfig.value("gpu", "unknown")},
                {"iterations_change", {
                    {"old", oldConfig.value("iterations", 0)},
                    {"new", newConfig.value("parameters", 0)}
                }},
                {"added_optimizations", false}
            }},
            {"configuration_summary", {
                {"has_gpu_optimizations", newConfig.contains("gpu_optimizations")},
                {"has_cpu_optimizations", newConfig.contains("cpu_optimizations")},
                {"auto_optimization_applied", newConfig.value("auto_optimize", false)}
            }}
        };
        
        std::string reportPath = newPath + ".migration_report.json";
        std::ofstream reportFile(reportPath);
        reportFile << report.dump(4);
        reportFile.close();
        
        std::cout << "Migration report saved to: " << reportPath << std::endl;
    }
    
    void generateBatchMigrationReport(const std::vector<std::string>& successful,
                                     const std::vector<std::string>& failed,
                                     const std::string& outputDirectory) {
        
        json batchReport = {
            {"batch_migration_report", true},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()},
            {"summary", {
                {"total_processed", static_cast<int>(successful.size() + failed.size())},
                {"successful", static_cast<int>(successful.size())},
                {"failed", static_cast<int>(failed.size())},
                {"success_rate", static_cast<double>(successful.size()) / (successful.size() + failed.size()) * 100.0}
            }},
            {"successful_migrations", successful},
            {"failed_migrations", failed}
        };
        
        std::string reportPath = outputDirectory + "/batch_migration_report.json";
        std::ofstream reportFile(reportPath);
        reportFile << batchReport.dump(4);
        reportFile.close();
        
        std::cout << "Batch migration report saved to: " << reportPath << std::endl;
    }
    
    void createDirectory(const std::string& path) {
        #ifdef _WIN32
        _mkdir(path.c_str());
        #else
        mkdir(path.c_str(), 0777);
        #endif
    }
    
    std::string generateUniquePath(const std::string& directory, const std::string& originalPath, const std::string& extension) {
        std::string filename = extractFilename(originalPath) + extension;
        std::string fullPath = directory + "/" + filename;
        
        // Add suffix if file already exists
        int suffix = 1;
        while (std::ifstream(fullPath).good()) {
            filename = extractFilename(originalPath) + "_" + std::to_string(suffix) + extension;
            fullPath = directory + "/" + filename;
            suffix++;
        }
        
        return fullPath;
    }
    
    std::string extractFilename(const std::string& path) {
        size_t slashPos = path.find_last_of("/\\");
        if (slashPos != std::string::npos) {
            return path.substr(slashPos + 1);
        }
        return path;
    }
};

int main() {
    try {
        ConfigurationMigrator migrator;
        
        std::string oldConfig = "old_config.json";
        std::string newConfig = "migrated_config.json";
        
        std::cout << "=== Configuration Migration Example ===" << std::endl;
        
        // Single file migration
        if (migrator.migrateConfiguration(oldConfig, newConfig)) {
            std::cout << "Configuration successfully migrated!" << std::endl;
        } else {
            std::cout << "Configuration migration failed!" << std::endl;
            return 1;
        }
        
        // Example batch migration
        std::vector<std::string> configFiles = {
            "config1.json",
            "config2.json", 
            "config3.json"
        };
        
        std::cout << "\n=== Batch Migration Example ===" << std::endl;
        if (migrator.batchMigrate(configFiles, "migrated_configs")) {
            std::cout << "Batch migration completed successfully!" << std::endl;
        } else {
            std::cout << "Batch migration completed with some failures!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Best Practices Summary

### CPU Backend Best Practices

1. **Memory Optimization**
   ```cpp
   // Use appropriate subimage size for CPU cache
   config.subimageSize = 256;  // Optimal for most CPUs
   
   // Enable FFTW optimization
   config.optimizePlans = true;
   
   // Monitor memory usage
   if (cpuAlgorithm->checkMemoryCriticalSystem()) {
       std::cout << "System memory critical, reducing subimage size" << std::endl;
       config.subimageSize = std::max(128, config.subimageSize / 2);
   }
   ```

2. **Performance Tuning**
   ```cpp
   // Optimal for multi-core CPUs
   config.grid = true;  // Enable subimage parallelization
   config.ompThreads = omp_get_max_threads();  // Use all available cores
   
   // Balance between quality and performance
   config.iterations = calculateOptimalIterations(imageSize);
   ```

3. **Error Handling**
   ```cpp
   try {
       auto algorithm = createAlgorithm("rl", "none");
       algorithm->configure(config);
       auto result = algorithm->run(data, psfs);
       
       // Validate output
       if (!validateOutput(result)) {
           throw std::runtime_error("Output validation failed");
       }
   } catch (const std::exception& e) {
       std::cerr << "CPU processing error: " << e.what() << std::endl;
       // Implement fallback or recovery
   }
   ```

### GPU Backend Best Practices

1. **Memory Management**
   ```cpp
   // Optimize for GPU memory
   config.subimageSize = 512;  // Larger sizes benefit GPU parallelism
   
   // Enable GPU-specific optimizations
   config.usePinnedMemory = true;     // Faster host-device transfers
   config.useAsyncTransfers = true;  // Overlap computation and transfers
   config.useCUBEKernels = true;     // Use optimized kernels
   
   // Check GPU memory availability
   if (!gpuAlgorithm->checkGPUMemoryAvailability(required_memory)) {
       config.subimageSize = std::max(256, config.subimageSize / 2);
   }
   ```

2. **Device Selection**
   ```cpp
   // Select optimal GPU device
   int optimalDevice = selectBestGPU();
   gpuAlgorithm->setGPUDevice(optimalDevice);
   
   // Monitor GPU performance
   gpuAlgorithm->resetGPUStats();
   auto start = std::chrono::high_resolution_clock::now();
   // ... execute algorithm ...
   auto end = std::chrono::high_resolution_clock::now();
   gpuAlgorithm->logPerformanceMetrics();
   ```

3. **Error Recovery**
   ```cpp
   try {
       // Attempt GPU processing
       if (config.gpu == "cuda") {
           auto gpuResult = processWithGPU(config);
           if (gpuResult.success) {
               return gpuResult;
           }
       }
       
       // Fallback to CPU on GPU failure
       std::cout << "GPU processing failed, falling back to CPU" << std::endl;
       auto cpuResult = processWithCPU(config);
       return cpuResult;
       
   } catch (const std::exception& e) {
       std::cerr << "Error in GPU processing: " << e.what() << std::endl;
       // Implement CPU fallback or error recovery
   }
   ```

### General Best Practices

1. **Configuration Management**
   ```cpp
   // Use environment-appropriate configuration
   DeconvolutionConfig getConfig() {
       auto config = DeconvolutionConfig();
       
       // Auto-detect optimal settings
       auto factory = DeconvolutionAlgorithmFactory::getInstance();
       if (factory.isGPUSupported()) {
           config.gpu = "auto";
           config.subimageSize = 512;  // GPU-optimized size
       } else {
           config.gpu = "none";
           config.subimageSize = 256;  // CPU-optimized size
       }
       
       return config;
   }
   ```

2. **Monitoring and Logging**
   ```cpp
   // Comprehensive performance monitoring
   void performProcessingWithMonitoring(DeconvolutionConfig config) {
       auto monitor = std::make_unique<PerformanceMonitor>();
       
       // Log pre-processing metrics
       logSystemState("pre_processing");
       
       // Execute processing
       auto result = dolphin.deconvolve(request);
       
       // Log post-processing metrics
       logSystemState("post_processing");
       
       // Generate performance report
       monitor->generatePerformanceReport(result, config);
   }
   ```

3. **Modular Design**
   ```cpp
   // Create reusable processor classes
   class DeconvolutionPipeline {
   public:
       DeconvolutionResult process(const std::string& imagePath,
                                  const std::string& psfPath,
                                  ProcessingParameters params);
       
       void registerPreProcessor(PreProcessor processor);
       void registerPostProcessor(PostProcessor processor);
       void registerErrorHandler(ErrorHandler handler);
   };
   
   // Usage
   auto pipeline = DeconvolutionPipeline();
   pipeline.registerErrorHandler(errorHandler);
   auto result = pipeline.process(image, psf, params);
   ```

This comprehensive usage examples document provides detailed guidance for effectively using the refactored DOLPHIN CPU/GPU architecture across various scenarios, from basic usage to advanced monitoring and error handling.