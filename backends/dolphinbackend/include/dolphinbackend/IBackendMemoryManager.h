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

#include <string>
#include <stdexcept>
#include <mutex>
#include <memory>
#include <condition_variable>
#include "ComplexData.h"

class BackendConfig;

// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())
struct MemoryTracking{
    // Memory management
    size_t maxMemorySize;
    size_t totalUsedMemory;
    std::mutex memoryMutex;
    std::condition_variable memoryCondition;
    MemoryTracking() : maxMemorySize(0), totalUsedMemory(0) {}
    MemoryTracking(size_t maxMemory) : maxMemorySize(maxMemory), totalUsedMemory(0){}
};
class IBackendMemoryManager{
public:
    // Data management - provide default implementations
    IBackendMemoryManager() = default;
    virtual ~IBackendMemoryManager() = default;
    
    /**
     * Get the device type of this memory manager
     * @return Device type string
     */
    virtual std::string getDeviceString() const noexcept {
        return "unknown";
    }
    
    // Synchronization - default implementation for non-async backends
    virtual void sync() {
        // Default no-op implementation for backends that don't need synchronization
    }

    virtual void init(const BackendConfig& config){

    }
    
    // Memory management initialization
    virtual void setMemoryLimit(size_t maxMemorySize = 0) {
        NOT_IMPLEMENTED(setMemoryLimit);
    }

    virtual void allocateMemoryOnDevice(ComplexData& data) const {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }

    virtual bool isOnDevice(void* data) const {
        NOT_IMPLEMENTED(isOnDevice);
    }
    // ipnut always cpudata
    virtual ComplexData copyDataToDevice(const ComplexData& srcdata) const {
        NOT_IMPLEMENTED(copyDataToDevice);
    }
    
    // move data to cpu and then run destBackend.copyDataToDevice
    virtual ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const {
        NOT_IMPLEMENTED(moveDataFromDevice);
    }
    
    virtual void memCopy(const ComplexData& srcData, ComplexData& destdata) const {
        NOT_IMPLEMENTED(memCopy);
    }
    
    virtual ComplexData copyData(const ComplexData& srcdata) const {
        NOT_IMPLEMENTED(copyData);
    }

   
    virtual ComplexData allocateMemoryOnDevice(const CuboidShape& shape) const {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }
    
    virtual void freeMemoryOnDevice(ComplexData& data) const {
        NOT_IMPLEMENTED(freeMemoryOnDevice);
    }

    virtual size_t getAvailableMemory() const {
        NOT_IMPLEMENTED(getAvailableMemory);
    }

    virtual size_t getAllocatedMemory() const {
        NOT_IMPLEMENTED(getAllocatedMemory);
    }



protected:
    mutable std::mutex backendMutex;

};
#undef NOT_IMPLEMENTED