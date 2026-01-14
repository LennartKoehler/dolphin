#pragma once
#include <dolphinbackend/IDeconvolutionBackend.h>
#include <dolphinbackend/IBackendMemoryManager.h>
#include "CUDABackend.h"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <array>
#include <thread>
#include <unordered_map>
#include <cuda_runtime.h>

// Forward declarations
class CUDABackendMemoryManager;
class CUDADeconvolutionBackend;
class CUDABackend;


/**
 * CUDABackendManager manages thread-specific CUDA backends directly.
 * This singleton class ensures each host thread gets its own CUDA backend,
 * providing clear ownership semantics and lifetime management.
 */
class CUDABackendManager {
public:
    struct ThreadBackend{
        std::shared_ptr<CUDABackend> backend;
        bool inUse;
    };
    /**
     * Get the singleton instance of CUDABackendManager
     * @return Reference to the singleton instance
     */
    static CUDABackendManager& getInstance();

    /**
     * Get or create a CUDA backend for the current thread
     * @return Shared pointer to a CUDABackend
     * @throws std::runtime_error if CUDA backend creation fails
     */
    std::shared_ptr<CUDABackend> getBackendForCurrentThread();

    /**
     * Release a CUDA backend (only valid for the thread that acquired it)
     * @param backend The backend to release
     */
    void releaseBackendForCurrentThread(CUDABackend*backend);

    /**
     * Set the maximum number of threads that can have backends
     * @param maxThreads Maximum number of threads to support
     */
    void setMaxThreads(size_t maxThreads);

    /**
     * Get the maximum number of threads supported
     * @return Maximum number of threads
     */
    size_t getMaxThreads() const;

    /**
     * Get the number of active thread backends
     * @return Number of active thread backends
     */
    size_t getActiveThreads() const;

    /**
     * Get the total number of backends created
     * @return Total number of backends
     */
    size_t getTotalBackends() const;

    /**
     * Clean up all backends and resources
     */
    void cleanup();

    // Delete copy constructor and assignment operator
    CUDABackendManager(const CUDABackendManager&) = delete;
    CUDABackendManager& operator=(const CUDABackendManager&) = delete;

private:
    /**
     * Private constructor for singleton pattern
     */
    CUDABackendManager(){
        cudaGetDeviceCount(&nDevices);
        int device;
        for (device = 0; device < nDevices; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            
            cudaSetDevice(device); //TODO overhead?

            size_t freeMem, totalMem;
            cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
            devices.push_back(CUDADevice{device, new MemoryTracking(totalMem)});

            printf("Device %d has compute capability %d.%d and %.2fGB memory\n",
            device, deviceProp.major, deviceProp.minor, (totalMem/1e9));
        }
        cudaSetDevice(0);
    };

    /**
     * Private destructor for singleton pattern
     */
    ~CUDABackendManager() = default;

    // Backend storage and management
    std::unordered_map<std::thread::id, ThreadBackend> threadBackends_;
    std::vector<CUDADevice> devices;

    // Threading synchronization
    mutable std::mutex managerMutex_;
    mutable std::condition_variable backendAvailable_;
    
    // Configuration
    int nDevices;
    int usedDeviceCounter = 0;
    size_t maxThreads_ = 32;  // Default maximum number of threads
    size_t totalCreated_ = 0; // Total backends created
    
    // Helper methods
    void initializeGlobalCUDA();
    std::shared_ptr<CUDABackend> createNewBackend();
    cudaStream_t createStream();
};
