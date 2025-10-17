#include "backend/BackendFactory.h"
#include "backend/Exceptions.h"
#include <iostream>
#include <dlfcn.h>

BackendFactory& BackendFactory::getInstance() {
    static BackendFactory instance;
    if (!instance.initialized_) {
        instance.registerBackends();
        instance.initialized_ = true;
    }
    return instance;
}

void BackendFactory::registerBackend(const std::string& name, BackendCreator creator) {
    if (backends_.find(name) != backends_.end()) {
        std::cerr << "[WARNING] Backend '" << name << "' is already registered. Overwriting." << std::endl;
    }
    backends_[name] = std::move(creator);
}

std::shared_ptr<IBackend> BackendFactory::create(const std::string& backendName) {
    auto deconv = createDeconvBackend(backendName);
    auto memory = createMemManager(backendName);
    
    return std::shared_ptr<IBackend>(new IBackend(backendName, deconv, memory));
}

std::shared_ptr<IBackendMemoryManager> BackendFactory::createMemManager(const std::string& backendName) {
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

std::shared_ptr<IDeconvolutionBackend> BackendFactory::createDeconvBackend(const std::string& backendName) {
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

std::vector<std::string> BackendFactory::getAvailableBackends() const {
    std::vector<std::string> names;
    names.reserve(backends_.size());
    
    for (const auto& [name, creator] : backends_) {
        names.push_back(name);
    }
    
    return names;
}

bool BackendFactory::isBackendAvailable(const std::string& name) const {
    return backends_.find(name) != backends_.end();
}

void BackendFactory::registerBackends() {
    std::cout << "[INFO] Registering backends..." << std::endl;
    
    // Register CPU backend
    registerBackend("cpu", []() {
        return std::shared_ptr<IBackend>(new IBackend("cpu", nullptr, nullptr));
    });
    
    // Register CUDA backend
    registerBackend("cuda", []() {
        return std::shared_ptr<IBackend>(new IBackend("cuda", nullptr, nullptr));
    });

    std::cout << "[INFO] Registered " << backends_.size() << " backend(s)" << std::endl;
}