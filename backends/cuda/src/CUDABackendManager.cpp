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

#include "CUDABackendManager.h"
#include <iostream>
#include <stdexcept>
#include "CUDABackend.h"
#include "dolphinbackend/Exceptions.h"




CUDABackendManager::CUDABackendManager(){
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


void CUDABackendManager::initializeGlobalCUDA() {
    // static bool globalInitialized = false;
    // static std::mutex initMutex;
    // static std::condition_variable initCondition;
    // static bool initInProgress = false;
    
    // std::unique_lock<std::mutex> lock(initMutex);
    // cudaStream_t stream1;
    // cudaStreamCreate(&stream1);

} 

std::shared_ptr<CUDABackend> CUDABackendManager::getNewBackendDifferentDevice(const CUDADevice& device){
    CUDADevice newdevice = devices[usedDeviceCounter];
    usedDeviceCounter = ++usedDeviceCounter % nDevices; // keep looping

    return createNewBackend(newdevice);    
}


std::shared_ptr<CUDABackend> CUDABackendManager::getNewBackendSameDevice(const CUDADevice& device){
    return createNewBackend(device);
}

// TODO make stream management more native, different streams in one cuda backend should not be permitted, one origin of truth
std::shared_ptr<CUDABackend> CUDABackendManager::createNewBackend(CUDADevice device) {
    // initializeGlobalCUDA();
    
    if (nDevices == 0) {
        throw dolphin::backend::BackendException(
            "No CUDA devices available", "CUDA", "createNewBackend");
    }
    
    if (devices.empty()) {
        throw dolphin::backend::BackendException(
            "No CUDA devices configured", "CUDA", "createNewBackend");
    }
    
    cudaError_t err = cudaSetDevice(device.id);
    CUDA_CHECK(err, "createNewBackend - cudaSetDevice");
    
    cudaStream_t stream;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CUDA_CHECK(err, "createNewBackend - cudaStreamCreateWithFlags");
    
    try {
        CUDABackend* backend = CUDABackend::create();
        if (!backend) {
            throw dolphin::backend::BackendException(
                "Failed to create CUDABackend", "CUDA", "createNewBackend");
        }
        
        backend->setStream(stream);
        backend->setDevice(device);
        


        // cudaDeviceSynchronize();
        CUDA_CHECK(err, "createNewBackend - cudaSetDevice reset");

        std::shared_ptr<CUDABackend> cudabackend = std::shared_ptr<CUDABackend>(backend);
        backends.push_back(cudabackend);
        return cudabackend;
    } catch (...) {
        // Clean up stream if backend creation fails
        cudaStreamDestroy(stream);
        throw;
    }
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
    

    
    g_logger(std::format("Cleaned up CUDA backend manager"), LogLevel::INFO);
}

cudaStream_t CUDABackendManager::createStream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CUDA_CHECK(err, "createStream - cudaStreamCreateWithFlags");
    return stream;
}




void CUDABackendManager::setMaxThreads(size_t maxThreads) {
    std::unique_lock<std::mutex> lock(managerMutex_);
    maxThreads_ = maxThreads;
}
