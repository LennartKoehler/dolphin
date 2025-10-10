#include "backend/BackendFactory.h"

std::shared_ptr<IBackend> BackendFactory::create(std::string backendName) {
    std::string libpath = std::string("backends/") + backendName + "/lib" + backendName + "_backend.so";
    void* handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        throw std::runtime_error(err ? err : "dlopen failed");
    }

    using create_deconv_fn = IDeconvolutionBackend*();
    auto create_backend_deconv = reinterpret_cast<create_deconv_fn*>(dlsym(handle, "createDeconvolutionBackend"));
    if (!create_backend_deconv) {
        throw std::runtime_error(dlerror());
    }
    std::shared_ptr<IDeconvolutionBackend> deconv = std::make_shared<IDeconvolutionBackend>(create_backend_deconv());

    using create_memory_fn = IBackendMemoryManager*();
    auto create_backend_memory = reinterpret_cast<create_memory_fn*>(dlsym(handle, "createBackendMemoryManager"));
    if (!create_backend_memory) {
        throw std::runtime_error(dlerror());
    }
    std::shared_ptr<IBackendMemoryManager> memory = std::make_shared<IBackendMemoryManager>(create_backend_memory());
    
    // Note: You'll need to determine the DeviceType based on backendName
    DeviceType deviceType = DeviceType::CPU; // Default, should be determined from backendName
    if (backendName == "cuda") {
        deviceType = DeviceType::CUDA;
    }
    
    return std::make_shared<IBackend>(deviceType, deconv, memory);
}

BackendFactory& BackendFactory::getInstance() {
    static BackendFactory instance;
    return instance;
}