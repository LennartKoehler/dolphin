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
#include "ComplexData.h"
// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

class IBackendMemoryManager{
public:
    // Data management - provide default implementations
    IBackendMemoryManager() = default;
    virtual ~IBackendMemoryManager() = default;
    
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