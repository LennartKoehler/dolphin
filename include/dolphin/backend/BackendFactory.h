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

#include <memory>
#include <string>
#include <stdexcept>
#include <map>
#include <functional>
#include <dlfcn.h>
#include <iostream>
#include "cpu_backend/CPUBackendManager.h"
#include "cuda_backend/CUDABackendManager.h"
#include <spdlog/spdlog.h>

// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

static std::function<void(const std::string&, LogLevel)> logCallback_fn = [](const std::string& msg, LogLevel level){
    switch(level){
    case LogLevel::INFO:
        spdlog::get("backend")->info(msg);
        break;
    case LogLevel::DEBUG:
        spdlog::get("backend")->debug(msg);
        break;
    case LogLevel::WARN:
        spdlog::get("backend")->warn(msg);
        break;
    case LogLevel::ERROR:
        spdlog::get("backend")->error(msg);
        break;
    default:
        break;
    } 
};
//helper
template <typename T>
inline constexpr bool always_false = false;
template <typename T>
[[noreturn]] T& unsupported_type() {
    static_assert(always_false<T>, "Unsupported type for getBackend");
}

#define DEFAULT_BACKEND "default"


struct BackendFactory {
    // ---------------- Singleton ----------------
    static BackendFactory& getInstance() {
        static BackendFactory instance;
        return instance;
    }

    // Access the default backend memory manager (single shared instance) // does this work
    IBackendMemoryManager& getDefaultBackendMemoryManager(){
        BackendConfig config{1, DEFAULT_BACKEND};
        static IBackendMemoryManager& mgr = getBackend<IBackendMemoryManager>(config);
        return mgr;
    }


    IBackendManager& getBackendManager(const std::string& backendName){
        IBackendManager* manager = findBackendManager(backendName);
        if (!manager) manager = loadBackendManager(backendName);
        if (!manager){
            spdlog::warn("Failed to load backend '{}', using default instead", backendName);
            return getBackendManager(DEFAULT_BACKEND);
        }
        assert(manager && "Couldnt even load default manager");
        return *manager;

    }

    // ---------------- Templated create ----------------
    template <typename T>
    T& getBackend(const BackendConfig& config) {
        T& b= loadTypedBackend<T>(config);
        return b;
    }



private:

    BackendFactory(){registerStaticBackends();} 
    ~BackendFactory() = default;
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;

    void registerStaticBackends(){
        
        addBackendManager("default", new CPUBackendManager());
        addBackendManager("cuda", new CUDABackendManager());
    }

    // ---------------- Internal loader ----------------
    template <typename T>
    T* loadSymbolFromLibrary(const std::string& backendName, const char* symbolName) {
        void* handle = getHandle(backendName);
        if (!handle) {
            // spdlog::warn("Could not load backend library '{}'", backendName);
            return nullptr;
        }

        using create_fn = T*();
        auto create_symbol = reinterpret_cast<create_fn*>(dlsym(handle, symbolName));
        if (!create_symbol) {
            dlclose(handle);
            return nullptr;
        }

        return create_symbol();
    }
    template <typename T>
    T& getBackend(IBackendManager& manager, const BackendConfig& config) {
        if constexpr (std::is_same_v<T, IBackend>) {
            return manager.getBackend(config);
        } else if constexpr (std::is_same_v<T, IBackendMemoryManager>) {
            return manager.getBackendMemoryManager(config);
        } else if constexpr (std::is_same_v<T, IDeconvolutionBackend>) {
            return manager.getDeconvolutionBackend(config);
        } else {
            static_assert(always_false<T>, "Unsupported interface type");
        }
    }




    template <typename T>
    T& loadTypedBackend(const BackendConfig& config) {
        const std::string& backendName = config.backendName;
        IBackendManager& manager = getBackendManager(config.backendName);

        return getBackend<T>(manager, config);
        

    }

    void addBackendManager(const std::string& backendName, IBackendManager* manager){
        manager->init(logCallback_fn);

        loadedManagers[backendName] = std::unique_ptr<IBackendManager>(manager);
    }

    IBackendManager* loadBackendManager(const std::string& backendName){
        IBackendManager* result = nullptr;
        // Try to load from library
        const char* symbolName = nullptr;
        symbolName = "createBackendManager";
        result = loadSymbolFromLibrary<IBackendManager>(backendName, symbolName);
    
        addBackendManager(backendName, result);
        return result;
    
    }

    IBackendManager* findBackendManager(const std::string& name){
        auto it = loadedManagers.find(name);
        if (it != loadedManagers.end()) return it->second.get();
        else return nullptr;
    }

    // ---------------- Shared library handle loader ----------------
    static void* getHandle(const std::string& backendName) {
        void* handle = dlopen(backendName.c_str(), RTLD_LAZY);
        if (!handle) {
            spdlog::warn("Failed to open library '{}': {}", backendName, dlerror());
        }
        return handle;
    }

    std::map<std::string, std::unique_ptr<IBackendManager>> loadedManagers;
};
