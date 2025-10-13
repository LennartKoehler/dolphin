#pragma once

#include <string>
#include <stdexcept>
#include <mutex>
#include "ComplexData.h"
// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

class IBackendMemoryManager{
public:
    // Data management - provide default implementations
    IBackendMemoryManager() = default;
    virtual ~IBackendMemoryManager() = default;

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
    
    virtual ComplexData allocateMemoryOnDevice(const RectangleShape& shape) const {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }
    
    virtual void freeMemoryOnDevice(ComplexData& data) const {
        NOT_IMPLEMENTED(freeMemoryOnDevice);
    }

    virtual size_t getAvailableMemory() const {
        NOT_IMPLEMENTED(getAvailableMemory);
    }

protected:
    mutable std::mutex backendMutex;

};
#undef NOT_IMPLEMENTED