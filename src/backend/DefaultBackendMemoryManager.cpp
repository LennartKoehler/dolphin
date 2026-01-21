#include "backend/DefaultBackendMemoryManager.h"
#include "dolphinbackend/Exceptions.h"
#include <memory>
#include <cstring>
#include <cstdlib>

// DefaultBackendMemoryManager implementation
DefaultBackendMemoryManager::DefaultBackendMemoryManager(){
}

DefaultBackendMemoryManager& DefaultBackendMemoryManager::getInstance(){
    static DefaultBackendMemoryManager instance;
    return instance;
}

DefaultBackendMemoryManager::~DefaultBackendMemoryManager() {

}



// DefaultBackendMemoryManager implementation
bool DefaultBackendMemoryManager::isOnDevice(void* ptr) const {
    // For Default backend, all valid pointers are "on device"
    return ptr != nullptr;
}

void DefaultBackendMemoryManager::allocateMemoryOnDevice(ComplexData& data) const {
    if (data.data != nullptr) {
        return; // Already allocated
    }
    size_t requested_size = sizeof(complex) * data.size.volume;
    data.data = (complex*)std::malloc(requested_size); 
    MEMORY_ALLOC_CHECK(data.data, requested_size, "Default", "allocateMemoryOnDevice");
    
    data.backend = this;
}

ComplexData DefaultBackendMemoryManager::allocateMemoryOnDevice(const RectangleShape& shape) const {
    ComplexData result{this, nullptr, shape};
    allocateMemoryOnDevice(result);
    return result;
}

ComplexData DefaultBackendMemoryManager::copyDataToDevice(const ComplexData& srcdata) const {
    ComplexData result = allocateMemoryOnDevice(srcdata.size);
    std::memcpy(result.data, srcdata.data, srcdata.size.volume * sizeof(complex));
    return result;
}

ComplexData DefaultBackendMemoryManager::moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const {
    BACKEND_CHECK(srcdata.data != nullptr, "Source data pointer is null", "Default", "moveDataFromDevice - source data");
    if (&destBackend == this){
        return srcdata;
    }
    else{
        // For cross-backend transfer, use the destination backend's copy method
        // since cpubackend is the "default" it is simple, be careful how this works for other backends though
        return destBackend.copyDataToDevice(srcdata);
    }
}

ComplexData DefaultBackendMemoryManager::copyData(const ComplexData& srcdata) const {
    BACKEND_CHECK(srcdata.data != nullptr, "Source data pointer is null", "Default", "copyData - source data");
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    memCopy(srcdata, destdata);
    return destdata;
}



void DefaultBackendMemoryManager::memCopy(const ComplexData& srcData, ComplexData& destData) const {
    BACKEND_CHECK(srcData.data != nullptr, "Source data pointer is null", "Default", "memCopy - source data");
    BACKEND_CHECK(destData.data != nullptr, "Destination data pointer is null", "Default", "memCopy - destination data");
    BACKEND_CHECK(destData.size.volume == srcData.size.volume, "Source and destination must have same size", "Default", "memCopy");
    std::memcpy(destData.data, srcData.data, srcData.size.volume * sizeof(complex));
}

void DefaultBackendMemoryManager::freeMemoryOnDevice(ComplexData& data) const {
    BACKEND_CHECK(data.data != nullptr, "Data pointer is null", "Default", "freeMemoryOnDevice - data pointer");
    size_t requested_size = sizeof(complex) * data.size.volume;
    std::free(data.data);
    
    data.data = nullptr;
}




