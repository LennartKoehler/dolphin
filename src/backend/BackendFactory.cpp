#include "backend/BackendFactory.h"

std::shared_ptr<IBackend> BackendFactory::create(std::string backendName) {
    auto deconv = createDeconvBackend(backendName);
    auto memory = createMemManager(backendName);
    
    return std::shared_ptr<IBackend>(new IBackend(backendName, deconv, memory));
}

std::shared_ptr<IBackendMemoryManager> BackendFactory::createMemManager(std::string backendName) {
    std::string libpath = std::string("backends/") + backendName + "/lib" + backendName + "_backend.so";
    void* handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        throw std::runtime_error(err ? err : "dlopen failed");
    }

    using create_memory_fn = IBackendMemoryManager*();
    auto create_backend_memory = reinterpret_cast<create_memory_fn*>(dlsym(handle, "createBackendMemoryManager"));
    if (!create_backend_memory) {
        throw std::runtime_error(dlerror());
    }
    
    return std::shared_ptr<IBackendMemoryManager>(create_backend_memory());
}

std::shared_ptr<IDeconvolutionBackend> BackendFactory::createDeconvBackend(std::string backendName) {
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
    
    return std::shared_ptr<IDeconvolutionBackend>(create_backend_deconv());
}

BackendFactory& BackendFactory::getInstance() {
    static BackendFactory instance;
    return instance;
}