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
#include "Exceptions.h"


struct BackendConfig;

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
    bool isAvailable(size_t amount){
        LockedAccess memory = LockedAccess(memoryMutex, data);
        if (memory.data.maxMemorySize > memory.data.totalUsedMemory + amount){
            return true;
        }
        return false;
    }
    bool allocate(size_t amount){
        LockedAccess memory = LockedAccess(memoryMutex, data);
        if (memory.data.maxMemorySize > memory.data.totalUsedMemory + amount){
            memory.data.totalUsedMemory += amount;
            return true;
        }
        return false;
    }
    bool deallocate(size_t amount){
        LockedAccess memory = LockedAccess(memoryMutex, data);
        if (amount <= memory.data.totalUsedMemory){
            memory.data.totalUsedMemory -= amount;
            return true;
        }
        return false;
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

    /**
     * True if this manager's memory is addressable from the host (CPU).
     * Used to reject direct element access on device buffers. Defaults to false
     * (treat unknown backends as device); host backends override to true.
     */
    virtual bool isHostBackend() const { return false; }

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




    /**
     * Handle-level query used by algorithms/IO: is this data reachable from this
     * backend?
     *
     * The check is based on the data's managing backend:
     * - Data from a host backend (CPU) is reachable iff this backend is also a
     *   host backend (host backends share one address space).
     * - Data from a device backend is reachable only from device backends; the
     *   pointer is then verified by this backend's raw isOnDevice() (e.g. CUDA
     *   inspects pointer attributes), which correctly handles io/worker managers
     *   sharing one physical device. A host backend never accepts device data —
     *   not every non-null pointer is host memory.
     */
    template<typename T>
    bool isOnDevice(const ManagedData<T>& data) const {
        const void* ptr = data.getData();
        if (ptr == nullptr) return false;
        const IBackendMemoryManager* db = data.getBackend();
        if (db == nullptr) return false;
        if (db->isHostBackend()) return isHostBackend();
        if (isHostBackend()) return false; // host backend asked about device data
        return isOnDevice(ptr);
    }

    // ============================================================================
    // Raw buffer access — backend-internal only.
    // ManagedData grants friendship to this interface, so the static helper
    // below can reach its (otherwise protected) getData(). Every concrete
    // backend obtains the raw pointer through this inherited helper, which
    // keeps the access path uniform and auditable. Static so that free-style
    // static member helpers in a backend can also call it.
    // ============================================================================
protected:
    // Pointer-level device query: backend-internal (used by the handle-level
    // isOnDevice above). Reasoning about raw pointers is reserved to the
    // backends; users query handles.
    virtual bool isOnDevice(const void* data) const {
        NOT_IMPLEMENTED(isOnDevice);
    }

    template<typename T>
    static T* rawData(const ManagedData<T>& data) { return data.getData(); }

    /**
     * Create a non-owning view over an existing buffer, tagged with this manager
     * as provenance (isOnDevice works on views). A view never frees its buffer;
     * ownership transfer to a view is done explicitly via transferOwnership().
     */
    template<typename T>
    DataView<T> createDataView(T* data, CuboidShape size, CuboidShape realSize, std::size_t bytes, std::size_t padding = 0) const {
        DataView<T> view{data, size, realSize, bytes, padding};
        view.backend = this; // provenance only; setBackend() on a view is a no-op
        return view;
    }

public:
    /**
     * Validated ownership transfer of the underlying buffer from a managed
     * handle to a DataView that must outlive it (e.g. a reinterpret view that is
     * returned while the allocating handle is dropped). Only ManagedData ->
     * DataView is possible by construction. Checks:
     * - the source must be backed by this backend and actually own its buffer
     *   (adopting foreign memory would free a non-backend pointer on
     *   destruction),
     * - the destination view must be backed by this backend,
     * - both handles must alias the same buffer.
     * On success the source is detached (won't free) and the result — a
     * ManagedData handle over the SAME buffer (same pointer, size, padding) —
     * owns the buffer. On failure nothing is modified and a BackendException is
     * thrown.
     */
    template<typename TFrom, typename TTo>
    ManagedData<TTo> transferOwnership(ManagedData<TFrom>& from, const DataView<TTo>& to) const {
        if (from.ownsMemory == false || from.getBackend() != this) {
            throw dolphin::backend::BackendException(
                "transferOwnership: source handle does not own a buffer of this backend",
                "dolphin", "transferOwnership",
                dolphin::backend::buildManagedDataContext(getDeviceString()));
        }
        if (to.getBackend() != this) {
            throw dolphin::backend::BackendException(
                "transferOwnership: destination view is not backed by this backend",
                "dolphin", "transferOwnership",
                dolphin::backend::buildManagedDataContext(getDeviceString()));
        }
        if (static_cast<const void*>(to.data) != static_cast<const void*>(from.data)) {
            throw dolphin::backend::BackendException(
                "transferOwnership: handles do not alias the same buffer",
                "dolphin", "transferOwnership",
                dolphin::backend::buildManagedDataContext(getDeviceString()));
        }
        from.releaseMemory();
        // Build the owning handle around the same buffer (no copy of the data):
        // a default construct + steal would invoke the deep-copy constructor.
        return ManagedData<TTo>(to.backend, to.data, to.getSize(), to.getRealSize(), to.getDataBytes(), to.getPadding());
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

    /**
     * Estimate the FFT library's internal workspace memory for the given shape.
     * This accounts for memory allocated by the FFT library (e.g. cuFFT) during
     * plan creation, separate from user-allocated data buffers.
     * @param shape The shape of the data to be transformed
     * @return Estimated workspace in bytes (0 if the backend has no FFT workspace overhead)
     */
    virtual float estimateFFTWorkspaceCopies(const CuboidShape& shape) const {
        return 0;
    }


};
