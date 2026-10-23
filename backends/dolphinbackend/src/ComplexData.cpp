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

#include "ComplexData.h"
#include "IBackendMemoryManager.h"

template<typename T>
ManagedData<T>::~ManagedData() {
    if (data) {
        try {
            if (ownsMemory && backend) {
                backend->freeMemoryOnDevice(*this);
            }
        } catch(...) {}
    }
}

template<typename T>
ManagedData<T>::ManagedData(const ManagedData& other)
    : backend(other.backend), size(other.size), realSize(other.realSize), padding(other.padding), ownsMemory(false) {
    // Copying an empty/backend-less handle yields an empty handle (guard: the
    // old code dereferenced a null backend here).
    if (backend == nullptr || other.data == nullptr) {
        this->backend = nullptr;
        this->data = nullptr;
        this->bytes = 0;
        return;
    }
    ManagedData copy = backend->createCopy(other);
    this->bytes = copy.bytes;
    this->data = copy.data;
    this->ownsMemory = true; // copy ctor steals the buffer from the temporary `copy`
    copy.data = nullptr; // Prevent copy's destructor from freeing the data
}

template<typename T>
ManagedData<T>& ManagedData<T>::operator=(const ManagedData& other) {
    if (this != &other) {
        // Free existing data
        if (data) {
            try {
                if (ownsMemory && backend) {
                    backend->freeMemoryOnDevice(*this);
                }
            } catch(...){}
        }

        // Assigning an empty/backend-less handle leaves this handle empty
        // (guard: the old code dereferenced a null backend here).
        if (other.backend == nullptr || other.data == nullptr) {
            data = nullptr;
            backend = nullptr;
            size = other.size;
            realSize = other.realSize;
            bytes = 0;
            padding = other.padding;
            ownsMemory = false;
            return *this;
        }

        // Copy from other
        backend = other.backend;
        size = other.size;
        realSize = other.realSize;
        padding = other.padding;
        ManagedData copy = backend->createCopy(other);
        data = copy.data;
        bytes = copy.bytes;
        ownsMemory = true; // copy-assign steals the buffer from the temporary `copy`
        copy.data = nullptr; // Prevent copy's destructor from freeing the data
    }
    return *this;
}

template<typename T>
ManagedData<T>::ManagedData(ManagedData&& other) noexcept
    : data(other.data), backend(other.backend), size(other.size), realSize(other.realSize), bytes(other.bytes),
      padding(other.padding), ownsMemory(other.ownsMemory) {
    other.data = nullptr;
    other.backend = nullptr;
    other.size = CuboidShape{};
    other.realSize = CuboidShape{};
    other.bytes = 0;
    other.padding = 0;
    other.ownsMemory = false;
}

template<typename T>
ManagedData<T>& ManagedData<T>::operator=(ManagedData&& other) noexcept {
    if (this != &other) {
        // Free existing data if any
        if (data) {
            try {
                if (ownsMemory && backend) {
                    backend->freeMemoryOnDevice(*this);
                }
            } catch(...){}
        }

        // Move data
        data = other.data;
        backend = other.backend;
        size = other.size;
        realSize = other.realSize;
        bytes = other.bytes;
        padding = other.padding;
        ownsMemory = other.ownsMemory;

        // Leave other in a valid state
        other.data = nullptr;
        other.backend = nullptr;
        other.size = CuboidShape{};
        other.realSize = CuboidShape{};
        other.bytes = 0;
        other.padding = 0;
        other.ownsMemory = false;
    }
    return *this;
}

template<typename T>
DataView<other_type_t<T>> ManagedData<T>::reinterpret() {
    return backend->reinterpret(*this);
}

template<typename T>
T& ManagedData<T>::access(std::size_t linearIndex){
    return data[convertIndex(linearIndex)];
}

template<typename T>
const T& ManagedData<T>::access(std::size_t linearIndex) const{
    return data[convertIndex(linearIndex)];
}

template<typename T>
size_t ManagedData<T>::convertIndex(size_t linearIndex) const {
    return linearIndex + (linearIndex / size.width) * padding;
}

// Explicit instantiations MUST come after all member definitions,
// otherwise the members defined above won't be instantiated.
template class ManagedData<real_t>;      // for RealData
template class ManagedData<complex_t>;   // for ComplexData
template class DataView<real_t>;      // for RealData
template class DataView<complex_t>;   // for ComplexData
