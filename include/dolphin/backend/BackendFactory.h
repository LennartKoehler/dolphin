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
#include "dolphinbackend/IBackend.h"
#include <dlfcn.h>
#include <iostream>
#include "cpu_backend/CPUBackend.h"
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

struct BackendFactory {
    // ---------------- Singleton ----------------
    static BackendFactory& getInstance() {
        static BackendFactory instance;
        (void)getDefaultBackendMemoryManager(); // initialize default memory manager (avoid copying non-copyable type)
        return instance;
    }

    // Access the default backend memory manager (single shared instance)
    static CPUBackendMemoryManager& getDefaultBackendMemoryManager();

    // ---------------- Templated create ----------------
    template <typename T>
    static std::shared_ptr<T> createShared(const BackendConfig& config) {
        T* raw = loadTypedBackend<T>(config);
        return std::shared_ptr<T>(raw, [](T* ptr){
            if(ptr) delete ptr;  // cleanup
        });
    }

    template <typename T>
    static std::unique_ptr<T> createUnique(const BackendConfig& config) {
        T* raw = loadTypedBackend<T>(config);
        return std::unique_ptr<T>(raw);  // unique_ptr uses delete by default
    }

private:

    BackendFactory() = default;
    ~BackendFactory() = default;
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;

    // ---------------- Internal loader ----------------
    template <typename T>
    static T* loadSymbolFromLibrary(const std::string& backendName, const char* symbolName) {
        void* handle = getHandle(backendName);
        if (!handle) {
            spdlog::warn("Could not load backend library '{}'", backendName);
            return nullptr;
        }

        using create_fn = T*();
        auto create_symbol = reinterpret_cast<create_fn*>(dlsym(handle, symbolName));
        if (!create_symbol) {
            spdlog::warn("Could not find symbol '{}' in backend library '{}'", symbolName, backendName);
            dlclose(handle);
            return nullptr;
        }

        return create_symbol();
    }

    template <typename T>
    static T* getBackend(IBackendManager* manager, BackendConfig config) {
        if constexpr (std::is_same_v<T, IBackend>) {
            return manager->getBackend(config)
        } else if constexpr (std::is_same_v<T, IBackendMemoryManager>) {
            return manager->getBackendMemoryManager(config);
        } else if constexpr (std::is_same_v<T, IDeconvolutionBackend>) {
            return manager->getDeconvolutionBackend(config);
        }
        return nullptr;
    }

    template <typename T>
    static T* loadTypedBackend(BackendConfig config) {
        const std::string& backendName = config.backendName;
        
        IBackendManager* manager = findBackendManager(config.backendName);
        if (!manager) addBackendManager(config.backendName);
        manager = findBackendManager(config.backendName);

        if (!manager) {
            spdlog::warn("Failed to load backend '{}', using default instead", backendName);
            manager = findBackendManager("default");
        }

        return getBackend<T>(manager, config);
        

    }

    void addBackendManager(const std::string& backendName){
        IBackendManager* result = nullptr;
        // Check if we should use the default backend
        if (backendName == "default" || backendName.empty()) {
            result = new CPUBackendManager();
        } else {
            // Try to load from library
            const char* symbolName = nullptr;
            symbolName = "createBackendManager";
            result = loadSymbolFromLibrary<IBackendManager>(backendName, symbolName);
        }    
    
        if (result) {            
            loadedManagers[backendName] = std::unique_ptr<IBackendManager>(result);
            result->setLogger(logCallback_fn);
        }
    
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

    std::unordered_map<std::string, std::unique_ptr<IBackendManager>> loadedManagers;
};
