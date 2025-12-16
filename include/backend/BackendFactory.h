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
#include "IBackend.h" 
#include <dlfcn.h>

// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())



class BackendFactory {
public:
    using BackendCreator = std::function<IBackend*()>;

    static std::shared_ptr<IBackend> createShared(const std::string& backendName);
    static std::unique_ptr<IBackend> createUnique(const std::string& backendName);
    static std::unique_ptr<IBackendMemoryManager> createMemManager(const std::string& backendName);
    static std::unique_ptr<IDeconvolutionBackend> createDeconvBackend(const std::string& backendName);

    static BackendFactory& getInstance();

    std::vector<std::string> getAvailableBackends() const;
    bool isBackendAvailable(const std::string& name) const;

    void registerBackend(const std::string& name, BackendCreator creator);



private:
    BackendFactory() = default;
    ~BackendFactory() = default;
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;

    IBackend* create(const std::string& backendName);
    void registerBackends();
    static void* getHandle(const std::string& backendName);

    std::map<std::string, BackendCreator> backends_;
    bool initialized_ = false;
};
