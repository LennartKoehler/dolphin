#include "CUDABackendManager.h"
#include <iostream>
#include <stdexcept>
#include "CUDABackend.h"
#include <dolphinbackend/Exceptions.h>
CUDABackendManager& CUDABackendManager::getInstance() {
    static CUDABackendManager instance;
    return instance;
}



void CUDABackendManager::initializeGlobalCUDA() {
    // static bool globalInitialized = false;
    // static std::mutex initMutex;
    // static std::condition_variable initCondition;
    // static bool initInProgress = false;
    
    // std::unique_lock<std::mutex> lock(initMutex);
    // cudaStream_t stream1;
    // cudaStreamCreate(&stream1);

} 


// TODO make stream management more native, different streams in one cuda backend should not be permitted, one origin of truth
std::shared_ptr<CUDABackend> CUDABackendManager::createNewBackend() {
    // initializeGlobalCUDA();


    
    CUDADevice device = devices[usedDeviceCounter];;
    usedDeviceCounter = ++usedDeviceCounter % nDevices; // keep looping

    cudaSetDevice(device.id);
    cudaStream_t stream = createStream();
    CUDABackend* backend = CUDABackend::create();
    backend->setStream(stream);
    backend->setDevice(device);
    cudaSetDevice(0);

    std::cout << "New CUDA Backend created" << usedDeviceCounter <<"   " << device.id << std::endl;

     return std::shared_ptr<CUDABackend>(backend);
}

std::shared_ptr<CUDABackend> CUDABackendManager::getBackendForCurrentThread() {
    std::unique_lock<std::mutex> lock(managerMutex_);
    
    std::thread::id currentThreadId = std::this_thread::get_id();

    // Check if we already have a backend for this thread
    auto it = threadBackends_.find(currentThreadId);
    if (it != threadBackends_.end() && !it->second.inUse) {
        it->second.inUse = true;
        return it->second.backend;
    }
    
    // Check if we can create a new backend
    if (threadBackends_.size() < maxThreads_) {
        auto backend = createNewBackend();
        threadBackends_[currentThreadId].backend = std::move(backend);
        totalCreated_++;

        threadBackends_[currentThreadId].inUse = true;
        return threadBackends_[currentThreadId].backend;
    }

    
    // Try again after waiting
    return getBackendForCurrentThread();
}

void CUDABackendManager::releaseBackendForCurrentThread(CUDABackend* backend) {
    std::unique_lock<std::mutex> lock(managerMutex_);
    
    std::thread::id currentThreadId = std::this_thread::get_id();
    
    // Check if this thread has an active backend
    auto it = threadBackends_.find(currentThreadId);
    if (it != threadBackends_.end() && it->second.inUse) {
        // Move the backend to the available pool
        it->second.inUse = false;
        
    } else {
        std::cerr << "[WARNING] Attempted to release backend for thread " << currentThreadId
                  << " that doesn't have an active backend" << std::endl;
    }
}

void CUDABackendManager::setMaxThreads(size_t maxThreads) {
    std::unique_lock<std::mutex> lock(managerMutex_);
    maxThreads_ = maxThreads;
}

size_t CUDABackendManager::getMaxThreads() const {
    std::unique_lock<std::mutex> lock(managerMutex_);
    return maxThreads_;
}

size_t CUDABackendManager::getActiveThreads() const {
    std::unique_lock<std::mutex> lock(managerMutex_);
    return threadBackends_.size();
}

size_t CUDABackendManager::getTotalBackends() const {
    std::unique_lock<std::mutex> lock(managerMutex_);
    return totalCreated_;
}

void CUDABackendManager::cleanup() {
    std::unique_lock<std::mutex> lock(managerMutex_);
    
    // Clean up all active thread backends
    for (auto& pair : threadBackends_) {
        if (pair.second.backend) {
            pair.second.backend->mutableDeconvManager().cleanup();
        }
    }
    threadBackends_.clear();
    

    
    std::cout << "[INFO] Cleaned up CUDA backend manager" << std::endl;
}

cudaStream_t CUDABackendManager::createStream() {
    cudaStream_t stream;
    // cudaError_t err = cudaStreamCreate(&stream);
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
    }
    return stream;
}
