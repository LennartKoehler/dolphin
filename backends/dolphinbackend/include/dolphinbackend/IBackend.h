#pragma once


#include <memory>
#include <string>
#include <stdexcept>
#include <functional>
#include "IComputeBackend.h"
#include "IBackendMemoryManager.h"


class IBackendManager;

enum LogLevel { DEBUG = 0, INFO, WARN, ERROR };

using LogCallback = std::function<void(const std::string& message, LogLevel level)>;

struct BackendConfig{

    size_t nThreads = 1; // whatever this means for the backendmanager. But manager has the opportunity to configure this
    std::string backendName = "default"; //TODO
};

class Owner{
public:
    // Constructor for external ownership (references to externally-owned components)
    Owner(IComputeBackend& compute, IBackendMemoryManager& mem)
        : computeBackend(nullptr), memoryBackend(nullptr) {}

    // Constructor for self-ownership (takes ownership of both components)
    Owner(std::unique_ptr<IComputeBackend> compute, std::unique_ptr<IBackendMemoryManager> mem)
        : computeBackend(std::move(compute)), memoryBackend(std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of compute, external memory)
    Owner(std::unique_ptr<IComputeBackend> compute, IBackendMemoryManager& mem)
        : computeBackend(std::move(compute)), memoryBackend(nullptr) {}

    bool ownsComputeBackend() const noexcept  {
        return computeBackend != nullptr;
    }

    bool ownsMemoryManager() const noexcept  {
        return memoryBackend != nullptr;
    }

    std::unique_ptr<IComputeBackend> releaseComputeBackend()  {
        if (!ownsComputeBackend()) {
            throw std::runtime_error("Cannot release compute backend: not owned by this Owner");
        }
        return std::move(computeBackend);
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager()  {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this Owner");
        }
        return std::move(memoryBackend);
    }

    void takeOwnership(std::unique_ptr<IComputeBackend> compute)  {
        computeBackend = std::move(compute);
    }

    void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem)  {
        memoryBackend = std::move(mem);
    }

private:
    std::unique_ptr<IComputeBackend> computeBackend;
    std::unique_ptr<IBackendMemoryManager> memoryBackend;
};
// Abstract interface for backend implementations
// Provides a common interface for different backend types (I, CUDA, etc.)
// to prevent mismatches between different backends the constructor is private
class IBackend {

    friend class IBackendManager;
    // friend class BackendFactory;
    // friend class IComputeBackend;
    // friend class IBackendMemoryManager;


public:
    virtual ~IBackend() = default;
    // Pure virtual methods that must be implemented by concrete backends
    virtual std::string getDeviceString() const noexcept = 0;


    // Ownership query methods
    virtual bool ownsComputeBackend() const noexcept = 0;
    virtual bool ownsMemoryManager() const noexcept = 0;
    virtual bool hasMemoryManager() const noexcept = 0;

    // Ownership transfer methods for both components
    virtual std::unique_ptr<IComputeBackend> releaseComputeBackend() = 0;
    virtual std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() = 0;
    virtual void takeOwnership(std::unique_ptr<IComputeBackend> compute) = 0;
    virtual void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) = 0;

    // Memory manager access - can be owned or shared depending on implementation
    virtual std::shared_ptr<IBackendMemoryManager> getSharedMemoryManager() const noexcept = 0;
    virtual IBackendMemoryManager* getMemoryManagerPtr() const noexcept = 0;

    // Access to the Owner object for advanced ownership management
    // virtual Owner<IComputeBackend, IBackendMemoryManager>& getOwner() noexcept = 0;
    // virtual const Owner<IComputeBackend, IBackendMemoryManager>& getOwner() const noexcept = 0;

    // Backend component access
    virtual const IComputeBackend& getComputeManager() const noexcept = 0;
    virtual const IBackendMemoryManager& getMemoryManager() const noexcept = 0;
    virtual IComputeBackend& mutableComputeManager() noexcept = 0;
    virtual IBackendMemoryManager& mutableMemoryManager() noexcept = 0;

    virtual void sync() = 0;



};
