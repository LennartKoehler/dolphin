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
#include <vector>

#include "ComplexData.h"


class BackendConfig;

// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

struct MemoryData {
    size_t maxMemorySize;
    size_t totalUsedMemory;

    MemoryData(size_t maxMemory = 0)
        : maxMemorySize(maxMemory), totalUsedMemory(0) {}
};
struct MemoryTracking {
private:
    MemoryData data;
    std::mutex memoryMutex;

    struct LockedAccess {
        std::unique_lock<std::mutex> lock;
        MemoryData& data;

        LockedAccess(std::mutex& m, MemoryData& d)
            : lock(m), data(d) {}
    };



public:
    MemoryTracking() : data(0) {}

    MemoryTracking(size_t maxMemory)
        : data(maxMemory) {}
    LockedAccess getAccess() {
        return LockedAccess(memoryMutex, data);
    }
};

class IBackendMemoryManager{
public:
    // Data management - provide default implementations
    IBackendMemoryManager() = default;
    virtual ~IBackendMemoryManager();

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

    // Memory management initialization
    virtual void setMemoryLimit(size_t maxMemorySize = 0) {
        NOT_IMPLEMENTED(setMemoryLimit);
    }

    template<typename T>
    void allocateMemoryOnDevice(ManagedData<T>& data) const {
        if (data.getData() != nullptr) {
            return; // Already allocated
        }

        size_t requested_size = data.getDataBytes();
        void* rawdata = allocateMemoryOnDevice(requested_size);
        data.setData((T*)rawdata);
    }

    virtual ComplexData allocateMemoryOnDeviceComplex(const CuboidShape& shape) const {
        NOT_IMPLEMENTED(allocateMemoryOnDeviceComplex);
    }


    virtual RealData allocateMemoryOnDeviceReal(const CuboidShape& shape) const{
        NOT_IMPLEMENTED(allocateMemoryOnDeviceReal);
    }

    virtual RealData allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const{
        NOT_IMPLEMENTED(allocateMemoryOnDeviceReal);
    }

    virtual ComplexData allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const{
        NOT_IMPLEMENTED(allocateMemoryOnDeviceComplexFull);
    }

    virtual void* allocateMemoryOnDevice(size_t) const;

    virtual DataView<real_t> reinterpret(ComplexData& data) const{
        NOT_IMPLEMENTED(reinterpret);
    }
    virtual DataView<complex_t> reinterpret(RealData& data) const{
        NOT_IMPLEMENTED(reinterpret);
    }




    virtual bool isOnDevice(const void* data) const {
        NOT_IMPLEMENTED(isOnDevice);
    }

    // ============================================================================
    // Low-level void* based memory operations (to be implemented by backends)
    // ============================================================================

    /**
     * Copy data from host to device
     * @param src Pointer to source data on host
     * @param size Size in bytes
     * @param shape Shape of the data
     * @return Pointer to allocated device memory
     */
    virtual void* copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const {
        NOT_IMPLEMENTED(copyDataToDevice);
    }


    /**
     * Move data from device to another backend
     * @param src Pointer to source data on device
     * @param size Size in bytes
     * @param shape Shape of the data
     * @param destBackend Destination backend
     * @return Pointer to allocated memory on destination backend
     */
    virtual void* moveDataFromDevice(void* src, size_t size, const CuboidShape& shape,
                                      const IBackendMemoryManager& destBackend) const {
        NOT_IMPLEMENTED(moveDataFromDevice);
    }

    /**
     * Memory copy between two pointers
     * @param src Pointer to source data
     * @param dest Pointer to destination data
     * @param size Size in bytes
     * @param shape Shape of the data
     */
    virtual void memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const {
        NOT_IMPLEMENTED(memCopy);
    }

    template<typename T>
    void memCopy(const ManagedData<T>& srcdata, const ManagedData<T>& destdata) const {
        assert(srcdata.getData() != nullptr);
        size_t byteSize = srcdata.getDataBytes();
        memCopy(srcdata.getData(), destdata.getData(), byteSize, srcdata.getSize());
    }

    template<typename T>
    T** createDataArray(std::vector<ManagedData<T>*>& data) const {
        int N = data.size();
        //TODO add check etc.
        size_t size = sizeof(void*) * N;
        T** dataPointer = (T**)malloc(size);
        for (int i = 0; i < N; ++i) {
            dataPointer[i] = data[i]->getData();
        }
        return dataPointer;
    }

    virtual void freeMemoryOnDevice(void* ptr, size_t size) const {
        NOT_IMPLEMENTED(freeMemoryOnDevice);
    }


    template<typename T>
    ManagedData<T> copyDataToDevice(const ManagedData<T>& srcdata) const {
        assert(srcdata.getData() != nullptr);
        size_t byteSize = srcdata.getDataBytes();
        void* result = copyDataToDevice(srcdata.getData(), byteSize, srcdata.getSize());
        return ManagedData<T>(this, static_cast<T*>(result), srcdata.getSize(), srcdata.getSize(), byteSize, srcdata.getPadding());
    }

    template<typename T>
    ManagedData<T> createCopy(const ManagedData<T>& srcdata) const {
        assert(srcdata.getData() != nullptr);
        size_t byteSize = srcdata.getDataBytes();
        void* result = allocateMemoryOnDevice(byteSize);
        memCopy(srcdata.getData(), result, byteSize, srcdata.getSize());
        return ManagedData<T>(this, static_cast<T*>(result), srcdata.getSize(), srcdata.getSize(), byteSize, srcdata.getPadding());
    }


    template<typename T>
    ManagedData<T> moveDataFromDevice(const ManagedData<T>& srcdata, const IBackendMemoryManager& destBackend) const {
        assert(srcdata.getData() != nullptr);
        size_t byteSize = srcdata.getDataBytes();
        void* result = moveDataFromDevice(srcdata.getData(), byteSize, srcdata.getSize(), destBackend);
        return ManagedData<T>(&destBackend, static_cast<T*>(result), srcdata.getSize(), srcdata.getSize(), srcdata.getDataBytes(), srcdata.getPadding());
    }


    template<typename T>
    void freeMemoryOnDevice(ManagedData<T>& data) const {
        if (data.getData() == nullptr) return;
        size_t byteSize = data.getDataBytes();
        freeMemoryOnDevice(data.getData(), byteSize);
        data.setData(nullptr);
    }

    virtual size_t getAvailableMemory() const {
        NOT_IMPLEMENTED(getAvailableMemory);
    }

    virtual size_t getAllocatedMemory() const {
        NOT_IMPLEMENTED(getAllocatedMemory);
    }


};
