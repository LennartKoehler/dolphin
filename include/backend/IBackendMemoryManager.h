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
    virtual void allocateMemoryOnDevice(ComplexData& data) {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }

    virtual bool isOnDevice(void* data) {
        NOT_IMPLEMENTED(isOnDevice);
    }
    
    virtual ComplexData moveDataToDevice(const ComplexData& srcdata) {
        NOT_IMPLEMENTED(moveDataToDevice);
    }
    
    virtual ComplexData moveDataFromDevice(const ComplexData& srcdata) {
        NOT_IMPLEMENTED(moveDataFromDevice);
    }
    
    virtual void memCopy(const ComplexData& srcData, ComplexData& destdata) {
        NOT_IMPLEMENTED(memCopy);
    }
    
    virtual ComplexData copyData(const ComplexData& srcdata) {
        NOT_IMPLEMENTED(copyData);
    }
    
    virtual ComplexData allocateMemoryOnDevice(const RectangleShape& shape) {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }
    
    virtual void freeMemoryOnDevice(ComplexData& data) {
        NOT_IMPLEMENTED(freeMemoryOnDevice);
    }

    virtual size_t getAvailableMemory() {
        NOT_IMPLEMENTED(getAvailableMemory);
    }

protected:
    std::mutex backendMutex;

};
#undef NOT_IMPLEMENTED