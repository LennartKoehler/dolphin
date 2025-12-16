/*
Copyright by Lennart Koehler
*/



#pragma once

#include "IDeconvolutionBackend.h"
#include "IBackendMemoryManager.h"
#include <memory>
#include <string>
#include <stdexcept>




class Owner{
public:
    // Constructor for external ownership (references to externally-owned components)
    Owner(IDeconvolutionBackend& deconv, IBackendMemoryManager& mem)
        : deconvBackend(nullptr), memoryBackend(nullptr) {}

    // Constructor for self-ownership (takes ownership of both components)
    Owner(std::unique_ptr<IDeconvolutionBackend> deconv, std::unique_ptr<IBackendMemoryManager> mem)
        : deconvBackend(std::move(deconv)), memoryBackend(std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of deconv, external memory)
    Owner(std::unique_ptr<IDeconvolutionBackend> deconv, IBackendMemoryManager& mem)
        : deconvBackend(std::move(deconv)), memoryBackend(nullptr) {}

    bool ownsDeconvBackend() const noexcept  {
        return deconvBackend != nullptr;
    }

    bool ownsMemoryManager() const noexcept  {
        return memoryBackend != nullptr;
    }

    std::unique_ptr<IDeconvolutionBackend> releaseDeconvBackend()  {
        if (!ownsDeconvBackend()) {
            throw std::runtime_error("Cannot release deconvolution backend: not owned by this Owner");
        }
        return std::move(deconvBackend);
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager()  {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this Owner");
        }
        return std::move(memoryBackend);
    }

    void takeOwnership(std::unique_ptr<IDeconvolutionBackend> deconv)  {
        deconvBackend = std::move(deconv);
    }

    void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem)  {
        memoryBackend = std::move(mem);
    }

private:
    std::unique_ptr<IDeconvolutionBackend> deconvBackend;
    std::unique_ptr<IBackendMemoryManager> memoryBackend;
};
// Abstract interface for backend implementations
// Provides a common interface for different backend types (I, CUDA, etc.)
// to prevent mismatches between different backends the constructor is private
class IBackend {

    friend class BackendFactory;
    friend class IDeconvolutionBackend;
    friend class IBackendMemoryManager;

public:
    // Abstract Owner class manages the actual lifetime of both deconvolution backend and memory manager


public:
    virtual ~IBackend(){

    }
    // Pure virtual methods that must be implemented by concrete backends
    virtual std::string getDeviceType() const noexcept = 0;
    
    // Ownership query methods
    virtual bool ownsDeconvolutionBackend() const noexcept = 0;
    virtual bool ownsMemoryManager() const noexcept = 0;
    virtual bool hasMemoryManager() const noexcept = 0;

    // Ownership transfer methods for both components
    virtual std::unique_ptr<IDeconvolutionBackend> releaseDeconvolutionBackend() = 0;
    virtual std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() = 0;
    virtual void takeOwnership(std::unique_ptr<IDeconvolutionBackend> deconv) = 0;
    virtual void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) = 0;

    // Memory manager access - can be owned or shared depending on implementation
    virtual std::shared_ptr<IBackendMemoryManager> getSharedMemoryManager() const noexcept = 0;
    virtual IBackendMemoryManager* getMemoryManagerPtr() const noexcept = 0;

    // Access to the Owner object for advanced ownership management
    // virtual Owner<IDeconvolutionBackend, IBackendMemoryManager>& getOwner() noexcept = 0;
    // virtual const Owner<IDeconvolutionBackend, IBackendMemoryManager>& getOwner() const noexcept = 0;

    // Backend component access
    virtual const IDeconvolutionBackend& getDeconvManager() const noexcept = 0;
    virtual const IBackendMemoryManager& getMemoryManager() const noexcept = 0;
    virtual IDeconvolutionBackend& mutableDeconvManager() noexcept = 0;
    virtual IBackendMemoryManager& mutableMemoryManager() noexcept = 0;

    // Clone method - creates a new thread-specific backend
    virtual std::shared_ptr<IBackend> onNewThread(std::shared_ptr<IBackend> original) const = 0;
    virtual void releaseBackend() = 0;
    virtual void sync() = 0;
    
};