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

// doesnt actually have a state though :)
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
    static std::shared_ptr<T> createShared(const std::string& backendName) {
        T* raw = loadTypedBackend<T>(backendName);
        return std::shared_ptr<T>(raw, [](T* ptr){
            if(ptr) delete ptr;  // cleanup
        });
    }

    template <typename T>
    static std::unique_ptr<T> createUnique(const std::string& backendName) {
        T* raw = loadTypedBackend<T>(backendName);
        return std::unique_ptr<T>(raw);  // unique_ptr uses delete by default
    }

private:

    BackendFactory() = default;
    ~BackendFactory() = default;
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;

    // ---------------- Internal loader ----------------
    template <typename T>
    static T* loadBackend(const std::string& backendName, const char* symbolName) {
        if (backendName == "default"){
            return nullptr;
        }
        void* handle = getHandle(backendName);
        if (!handle) {
            spdlog::warn("Could not load backend library '{}', using default instead", backendName);
            return nullptr;
        }

        using setlogger_fn = void(LogCallback fn);
        auto setlogger = reinterpret_cast<setlogger_fn*>(dlsym(handle, "set_backend_logger"));
        if (!setlogger) {
            spdlog::warn("Could not find symbol '{}' in backend library '{}', using default instead", symbolName, backendName);
            dlclose(handle);
            return nullptr;
        }

        setlogger(logCallback_fn);
        using create_fn = T*();
        auto create_backend = reinterpret_cast<create_fn*>(dlsym(handle, symbolName));
        if (!create_backend) {
            spdlog::warn("Could not find symbol '{}' in backend library '{}', using default instead", symbolName, backendName);
            dlclose(handle);
            return nullptr;
        }

        return create_backend();
    }

    template <typename T>
    static T* loadTypedBackend(const std::string& backendName) {
        T* result;
        if constexpr (std::is_same_v<T, IBackend>) {
            result = loadBackend<IBackend>(backendName, "createBackend");
            if (result == nullptr) result = CPUBackend::create();
        } else if constexpr (std::is_same_v<T, IBackendMemoryManager>) {
            result = loadBackend<IBackendMemoryManager>(backendName, "createBackendMemoryManager");
            if (result == nullptr) result = new CPUBackendMemoryManager();
        } else if constexpr (std::is_same_v<T, IDeconvolutionBackend>) {
            result = loadBackend<IDeconvolutionBackend>(backendName, "createDeconvolutionBackend");
            if (result == nullptr) result = new CPUDeconvolutionBackend();
        }
        return result;
    }


    // ---------------- Shared library handle loader ----------------
    static void* getHandle(const std::string& backendName) {
        void* handle = dlopen(backendName.c_str(), RTLD_LAZY);
        if (!handle) {
            spdlog::warn("Failed to open library '{}': {}", backendName, dlerror());
        }
        return handle;
    }
};
