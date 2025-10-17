#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <map>
#include <functional>
#include "IDeconvolutionBackend.h"
#include "IBackendMemoryManager.h"
#include <dlfcn.h>


class IBackend {
    friend class BackendFactory;  // only factory can construct

private:
    std::shared_ptr<IDeconvolutionBackend> deconvManager;
    std::shared_ptr<IBackendMemoryManager> memoryManager;
    std::string deviceType;

    // Private constructor — only accessible by factory
    IBackend(std::string type,
             std::shared_ptr<IDeconvolutionBackend> deconv,
             std::shared_ptr<IBackendMemoryManager> mem)
        : deconvManager(std::move(deconv)),
          memoryManager(std::move(mem)),
          deviceType(type) {}

public:
    std::string getDeviceType() const noexcept { return deviceType; }

    const IDeconvolutionBackend& getDeconvManager() const noexcept {
        return *deconvManager;
    }

    const IBackendMemoryManager& getMemoryManager() const noexcept {
        return *memoryManager;
    }

    // Optionally, allow non-const access if you need modification
    IDeconvolutionBackend& mutableDeconvManager() noexcept {
        return *deconvManager;
    }

    IBackendMemoryManager& mutableMemoryManager() noexcept {
        return *memoryManager;
    }
};

class BackendFactory {
public:
    using BackendCreator = std::function<std::shared_ptr<IBackend>()>;

    static std::shared_ptr<IBackend> create(const std::string& backendName);
    static std::shared_ptr<IBackendMemoryManager> createMemManager(const std::string& backendName);
    static std::shared_ptr<IDeconvolutionBackend> createDeconvBackend(const std::string& backendName);

    static BackendFactory& getInstance();

    std::vector<std::string> getAvailableBackends() const;
    bool isBackendAvailable(const std::string& name) const;

    void registerBackend(const std::string& name, BackendCreator creator);

private:
    BackendFactory() = default;
    ~BackendFactory() = default;
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;

    void registerBackends();

    std::map<std::string, BackendCreator> backends_;
    bool initialized_ = false;
};
