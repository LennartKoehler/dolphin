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

#include "cuda_backend/CUDABackendManager.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <stdexcept>
#include "cuda_backend/CUDABackend.h"
#include "dolphinbackend/Exceptions.h"
#include <spdlog/fmt/fmt.h>



extern LogCallback g_logger_cuda;

void CUDABackendManager::init(LogCallback fn) {

    std::lock_guard<std::mutex> lock(mutex_);
    g_logger_cuda = std::move(fn);


    cudaError_t err = cudaGetDeviceCount(&nDevices);
    CUDA_CHECK(err, "cudaGetDeviceCount", buildCudaContext({CUDADevice{0, nullptr}, cudaStreamLegacy}));

    if (nDevices <= 0) {
        throw dolphin::backend::BackendException(
            "No CUDA devices found", "CUDA", "CUDABackendManager constructor",
            buildCudaContext({CUDADevice{0, nullptr}, cudaStreamLegacy}));
    }

    int device;
    // int nDevices = 1;
    for (device = 0; device < nDevices; ++device) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, device);
        CUDA_CHECK(err, "cudaGetDeviceProperties", buildCudaContext({CUDADevice{device, nullptr}, cudaStreamLegacy}));



        err = cudaSetDevice(device);
        CUDA_CHECK(err, "cudaSetDevice", buildCudaContext({CUDADevice{device, nullptr}, cudaStreamLegacy}));

        size_t freeMem, totalMem;
        err = cudaMemGetInfo(&freeMem, &totalMem);
        CUDA_CHECK(err, "cudaMemGetInfo", buildCudaContext({CUDADevice{device, nullptr}, cudaStreamLegacy}));

        if (totalMem == 0) {
            throw dolphin::backend::BackendException(
                "Device " + std::to_string(device) + " reports zero memory",
                "CUDA", "CUDABackendManager constructor",
                buildCudaContext({CUDADevice{device, nullptr}, cudaStreamLegacy}));
        }

        devices.push_back(CUDADevice{device, new MemoryTracking(totalMem)});

        std::ostringstream ctx;
        ctx << "cuda:cuda" << device << ":tid:" << std::this_thread::get_id() << ":stream:n/a";
        g_logger_cuda(
            ctx.str(),
            fmt::format("Device {} has compute capability {}.{} and {:.2f} GB memory", device, deviceProp.major, deviceProp.minor, (totalMem/1e9)),
            LogLevel::INFO);
        // printf("Device %d has compute capability %d.%d and %.2fGB memory\n",
        // device, deviceProp.major, deviceProp.minor, (totalMem/1e9));

    }

    // Reset to device 0 for default operations
    err = cudaSetDevice(0);
    CUDA_CHECK(err, "cudaSetDevice", buildCudaContext({CUDADevice{0, nullptr}, cudaStreamLegacy}));


}

extern "C" IBackendManager* createBackendManager() {
    return new CUDABackendManager();
}




CUDABackendManager::CUDABackendManager(){

};

int CUDABackendManager::getNumberDevices() const {
    return nDevices;
}

void CUDABackendManager::setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig){

}

// IComputeBackend& CUDABackendManager::getComputeBackend(const BackendConfig& config) {
//     auto compute = createComputeBackend(configToConfig(config));
//     std::unique_lock<std::mutex> lock(mutex_);
//     computeBackends.push_back(std::move(compute));
//     return *computeBackends.back();
// }
//
// IBackendMemoryManager& CUDABackendManager::getBackendMemoryManager(const BackendConfig& config) {
//     auto manager = createMemoryManager(configToConfig(config));
//     std::unique_lock<std::mutex> lock(mutex_);
//     memoryManagers.push_back(std::move(manager));
//     return *memoryManagers.back();
// }

IBackend& CUDABackendManager::createBackendForCurrentThread(const BackendConfig& config) {
    CUDABackendConfig cudaconfig = configToConfig(config);
    // auto compute = createComputeBackend(cudaconfig);
    // auto mem = createMemoryManager(cudaconfig);
    // auto backend = std::unique_ptr<CUDABackend>(new CUDABackend(cudaconfig, std::move(compute), std::move(mem)));
    // std::unique_lock<std::mutex> lock(mutex_);
    // IBackend& ref = *backend;
    // backends.push_back(std::move(backend));
    CUDABackend& backend = createNewBackend(cudaconfig);
    return backend;
}

CUDABackendConfig CUDABackendManager::configToConfig(const BackendConfig& config) {

    CUDADevice device = devices[usedDeviceCounter];
    usedDeviceCounter = ++usedDeviceCounter % nDevices; // keep looping

    CUDABackendConfig cudaconfig{device, createStream()};
    return cudaconfig;
}


// IBackend& CUDABackendManager::clone(IBackend& backend, const BackendConfig& config){cudabacked
//
//     CUDADevice newdevice = devices[usedDeviceCounter];
//     usedDeviceCounter = ++usedDeviceCounter % nDevices; // keep looping
//
//     CUDABackendConfig cudaconfig{newdevice, 0};
//     return createNewBackend(cudaconfig);
// }


IBackend& CUDABackendManager::createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config){
    CUDABackend& backend_ = dynamic_cast<CUDABackend&>(backend);
    return createNewBackend(backend_.config);
}

// TODO make stream management more native, different streams in one cuda backend should not be permitted, one origin of truth
std::unique_ptr<CUDAComputeBackend> CUDABackendManager::createComputeBackend(CUDABackendConfig config) {
    return std::make_unique<CUDAComputeBackend>(config);
}

std::unique_ptr<CUDABackendMemoryManager> CUDABackendManager::createMemoryManager(CUDABackendConfig config) {
    return std::make_unique<CUDABackendMemoryManager>(config);
}

CUDABackend& CUDABackendManager::createNewBackend(CUDABackendConfig config) {
    // initializeGlobalCUDA();

    if (nDevices == 0) {
        throw dolphin::backend::BackendException(
            "No CUDA devices available", "CUDA", "createNewBackend", buildCudaContext(config));
    }

    if (devices.empty()) {
        throw dolphin::backend::BackendException(
            "No CUDA devices configured", "CUDA", "createNewBackend", buildCudaContext(config));
    }

    cudaError_t err = cudaSetDevice(config.device.id);
    CUDA_CHECK(err, "createNewBackend - cudaSetDevice", buildCudaContext(config));

    err = cudaStreamCreateWithFlags(&config.stream, cudaStreamNonBlocking);
    CUDA_CHECK(err, "createNewBackend - cudaStreamCreateWithFlags", buildCudaContext(config));

    try {
        auto compute = createComputeBackend(config);
        auto mem = createMemoryManager(config);
        std::unique_ptr<CUDABackend> backend = std::unique_ptr<CUDABackend>(new CUDABackend(config, std::move(compute), std::move(mem)));
        if (!backend) {
            throw dolphin::backend::BackendException(
                "Failed to create CUDABackend", "CUDA", "createNewBackend", buildCudaContext(config));
        }



        // cudaDeviceSynchronize();
        CUDA_CHECK(err, "createNewBackend - cudaSetDevice reset", buildCudaContext(config));

        backends.push_back(std::move(backend));
        return *backends.back();
    } catch (...) {
        // Clean up stream if backend creation fails
        cudaStreamDestroy(config.stream);
        throw;
    }
}




// void CUDABackendManager::cleanup() {
//     std::unique_lock<std::mutex> lock(mutex_);

//     // Clean up all active thread backends
//     for (auto& pair : threadBackends_) {
//         if (pair.second.backend) {
//             pair.second.backend->mutableComputeManager().cleanup();
//         }
//     }
//     threadBackends_.clear();



//     g_logger_cuda(fmt::format("Cleaned up CUDA backend manager"), LogLevel::INFO);
// }

cudaStream_t CUDABackendManager::createStream() const {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CUDA_CHECK(err, "createStream - cudaStreamCreateWithFlags", buildCudaContext({CUDADevice{0, nullptr}, cudaStreamLegacy}));
    return stream;
}




