/*
Copyright by Lennart Koehler
*/



#pragma once

#include "IDeconvolutionBackend.h"
#include "IBackendMemoryManager.h"
#include <memory>
#include <string>
#include <stdexcept>


// Abstract interface for backend implementations
// Provides a common interface for different backend types (CPU, CUDA, etc.)
// to prevent mismatches between different backends the constructor is private
class IBackend {

    friend class BackendFactory;
    friend class IDeconvolutionBackend;
    friend class IBackendMemoryManager;

public:
    // Owner class manages the actual lifetime of both deconvolution backend and memory manager
    class Owner {
    public:
        Owner() = default;
        
        // Constructor that takes ownership of deconvolution backend only
        explicit Owner(std::unique_ptr<IDeconvolutionBackend> deconv)
            : deconvBackend(std::move(deconv)) {}

        // Constructor that takes ownership of both deconvolution backend and memory manager
        Owner(std::unique_ptr<IDeconvolutionBackend> deconv, std::unique_ptr<IBackendMemoryManager> mem)
            : deconvBackend(std::move(deconv)), memoryManager(std::move(mem)) {}

        // Move constructor and assignment
        Owner(Owner&& other) noexcept
            : deconvBackend(std::move(other.deconvBackend)),
              memoryManager(std::move(other.memoryManager)) {}

        Owner& operator=(Owner&& other) noexcept {
            if (this != &other) {
                deconvBackend = std::move(other.deconvBackend);
                memoryManager = std::move(other.memoryManager);
            }
            return *this;
        }

        // Delete copy operations to prevent accidental copying
        Owner(const Owner&) = delete;
        Owner& operator=(const Owner&) = delete;

        // Access methods
        IDeconvolutionBackend* getDeconvBackend() const noexcept {
            return deconvBackend.get();
        }

        IBackendMemoryManager* getMemoryManager() const noexcept {
            return memoryManager.get();
        }

        // Check if this owner actually owns the components
        bool ownsDeconvBackend() const noexcept {
            return deconvBackend != nullptr;
        }

        bool ownsMemoryManager() const noexcept {
            return memoryManager != nullptr;
        }

        // Release ownership (returns the unique_ptr, transferring ownership)
        std::unique_ptr<IDeconvolutionBackend> releaseDeconvBackend() noexcept {
            return std::move(deconvBackend);
        }

        std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() noexcept {
            return std::move(memoryManager);
        }

        // Take ownership
        void takeOwnership(std::unique_ptr<IDeconvolutionBackend> deconv) {
            deconvBackend = std::move(deconv);
        }

        void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) {
            memoryManager = std::move(mem);
        }

    private:
        std::unique_ptr<IDeconvolutionBackend> deconvBackend;
        std::unique_ptr<IBackendMemoryManager> memoryManager;
    };

public:
    virtual ~IBackend() = default;

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
    virtual Owner& getOwner() noexcept = 0;
    virtual const Owner& getOwner() const noexcept = 0;

    // Backend component access
    virtual const IDeconvolutionBackend& getDeconvManager() const noexcept = 0;
    virtual const IBackendMemoryManager& getMemoryManager() const noexcept = 0;
    virtual IDeconvolutionBackend& mutableDeconvManager() noexcept = 0;
    virtual IBackendMemoryManager& mutableMemoryManager() noexcept = 0;

    // Clone method - creates a new thread-specific backend
    virtual std::shared_ptr<IBackend> onNewThread() const = 0;
    virtual void releaseBackend() {}
    virtual void sync() = 0;
    // Factory methods for different ownership models (pure virtual)
    // These create new instances with specific ownership patterns
    virtual std::shared_ptr<IBackend> createWithExternalOwnership(
        IDeconvolutionBackend& deconv,
        IBackendMemoryManager& mem) const = 0;

    virtual std::shared_ptr<IBackend> createWithSelfOwnership(
        std::unique_ptr<IDeconvolutionBackend> deconv,
        std::unique_ptr<IBackendMemoryManager> mem) const = 0;

    virtual std::shared_ptr<IBackend> createWithMixedOwnership(
        std::unique_ptr<IDeconvolutionBackend> deconv,
        IBackendMemoryManager& mem) const = 0;
};

