/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once
#include "dolphinbackend/IDeconvolutionBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
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


    std::shared_ptr<CUDABackend> getNewBackendDifferentDevice(const CUDADevice& device);
    std::shared_ptr<CUDABackend> getNewBackendSameDevice(const CUDADevice& device);

    std::shared_ptr<CUDABackend> getBackendForCurrentThread();

    std::shared_ptr<CUDABackend> getBackendForCurrentThreadSameDevice(CUDADevice device);

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
        cudaError_t err = cudaGetDeviceCount(&nDevices);
        CUDA_CHECK(err, "cudaGetDeviceCount");
        
        if (nDevices <= 0) {
            throw dolphin::backend::BackendException(
                "No CUDA devices found", "CUDA", "CUDABackendManager constructor");
        }
        
        int device;
        // int nDevices = 1;
        for (device = 0; device < nDevices; ++device) {
            cudaDeviceProp deviceProp;
            err = cudaGetDeviceProperties(&deviceProp, device);
            CUDA_CHECK(err, "cudaGetDeviceProperties");
            
 
            
            err = cudaSetDevice(device);
            CUDA_CHECK(err, "cudaSetDevice");

            size_t freeMem, totalMem;
            err = cudaMemGetInfo(&freeMem, &totalMem);
            CUDA_CHECK(err, "cudaMemGetInfo");
            
            if (totalMem == 0) {
                throw dolphin::backend::BackendException(
                    "Device " + std::to_string(device) + " reports zero memory",
                    "CUDA", "CUDABackendManager constructor");
            }
            
            devices.push_back(CUDADevice{device, new MemoryTracking(totalMem)});

            g_logger(std::format("Device {} has compute capability {}.{} and {:.2f} GB memory", device, deviceProp.major, deviceProp.minor, (totalMem/1e9)), LogLevel::INFO);
            // printf("Device %d has compute capability %d.%d and %.2fGB memory\n",
            // device, deviceProp.major, deviceProp.minor, (totalMem/1e9));
            
        }
        
        // Reset to device 0 for default operations
        err = cudaSetDevice(0);
        CUDA_CHECK(err, "cudaSetDevice");
    };

    /**
     * Private destructor for singleton pattern
     */
    ~CUDABackendManager() = default;

    // Backend storage and management

    std::unordered_map<std::thread::id, ThreadBackend> threadBackends_;
    std::vector<CUDADevice> devices;
    std::vector<std::shared_ptr<CUDABackend>> backends;

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
    std::shared_ptr<CUDABackend> createNewBackend(CUDADevice device);
    cudaStream_t createStream();
};
