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



std::shared_ptr<IBackend> BackendFactory::createShared(const std::string& backendName) {
    auto& factory = getInstance();
    auto it = factory.backends_.find(backendName);
    
    if (it != factory.backends_.end()) {
        return std::shared_ptr<IBackend>(it->second());
    }
    
    throw dolphin::backend::BackendException("Backend '" + backendName + "' not found", backendName, "createShared");
}

std::unique_ptr<IBackend> BackendFactory::createUnique(const std::string& backendName) {
    auto& factory = getInstance();
    auto it = factory.backends_.find(backendName);
    
    if (it != factory.backends_.end()) {
        return std::unique_ptr<IBackend>(it->second());
    }
    
    throw dolphin::backend::BackendException("Backend '" + backendName + "' not found", backendName, "createUnique");
}

void* BackendFactory::getHandle(const std::string& backendName){
    std::string libpath = std::string("backends/") + backendName + "/lib" + backendName + "_backend.so";
    void* handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        std::cerr << "[WARNING] Could not load backend library '" << backendName << "': " << (err ? err : "dlopen failed") << std::endl;
        return nullptr;
    }

    return handle;
}

std::unique_ptr<IBackendMemoryManager> BackendFactory::createMemManager(const std::string& backendName) {
    void* handle = getHandle(backendName);
    using create_memory_fn = IBackendMemoryManager*();
    auto create_backend_memory = reinterpret_cast<create_memory_fn*>(dlsym(handle, "createBackendMemoryManager"));
    if (!create_backend_memory) {
        std::cerr << "[WARNING] Could not find createBackendMemoryManager symbol in backend library '" << backendName << "'" << std::endl;
        dlclose(handle);
        return nullptr;
    }
    
    return std::unique_ptr<IBackendMemoryManager>(create_backend_memory());
}

std::unique_ptr<IDeconvolutionBackend> BackendFactory::createDeconvBackend(const std::string& backendName) {
    void* handle = getHandle(backendName);
    using create_deconv_fn = IDeconvolutionBackend*();
    auto create_backend_deconv = reinterpret_cast<create_deconv_fn*>(dlsym(handle, "createDeconvolutionBackend"));
    if (!create_backend_deconv) {
        std::cerr << "[WARNING] Could not find createDeconvolutionBackend symbol in backend library '" << backendName << "'" << std::endl;
        dlclose(handle);
        return nullptr;
    }
    
    return std::unique_ptr<IDeconvolutionBackend>(create_backend_deconv());
}





IBackend* BackendFactory::create(const std::string& backendName){
    void* handle = getHandle(backendName);
    using create_backend_fn = IBackend*();
    auto create_backend = reinterpret_cast<create_backend_fn*>(dlsym(handle, "createBackend"));
    if (!create_backend) {
        std::cerr << "[WARNING] Could not find createDBackend symbol in backend library '" << backendName << "'" << std::endl;
        dlclose(handle);
        return nullptr;
    }
    
    return create_backend();
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
    registerBackend("cpu", [this]() -> IBackend* {
        // auto deconv = createDeconvBackend("cpu");
        // auto memory = createMemManager("cpu");
        return create("cpu");
    });
    
    // Register OpenMP backend
    registerBackend("openmp", [this]() -> IBackend* {
        // auto deconv = createDeconvBackend("openmp");
        // auto memory = createMemManager("openmp");
        return create("openmp");
    });
    
    // Register CUDA backend
    registerBackend("cuda", [this]() -> IBackend* {
        // auto deconv = createDeconvBackend("cuda");
        // auto memory = createMemManager("cuda");
        return create("cuda");
    });

    std::cout << "[INFO] Registered " << backends_.size() << " backend(s)" << std::endl;
}