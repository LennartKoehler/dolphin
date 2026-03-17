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

template class ManagedData<real_t>;      // for RealData
template class ManagedData<complex_t>;   // for ComplexData

template<typename T>
ManagedData<T>::ManagedData(const IBackendMemoryManager* b, T* d, CuboidShape s)
    : backend(b), data(d), size(s) {}

template<typename T>
ManagedData<T>::~ManagedData() {
    if (data) {
        try {
            if (backend) {
                backend->freeMemoryOnDevice(*this);
            }
        } catch(...) {}
    }
}

template<typename T>
ManagedData<T>::ManagedData(const ManagedData& other)
    : backend(other.backend), size(other.size) {
    ManagedData copy = backend->createCopy(other);
    this->data = copy.data;
    copy.data = nullptr; // Prevent copy's destructor from freeing the data
}

template<typename T>
ManagedData<T>& ManagedData<T>::operator=(const ManagedData& other) {
    if (this != &other) {
        // Free existing data
        if (data) {
            try {
                if (backend) {
                    backend->freeMemoryOnDevice(*this);
                }
            } catch(...){}
        }

        // Copy from other
        backend = other.backend;
        size = other.size;
        ManagedData copy = backend->createCopy(other);
        data = copy.data;
        copy.data = nullptr; // Prevent copy's destructor from freeing the data
    }
    return *this;
}

template<typename T>
ManagedData<T>::ManagedData(ManagedData&& other) noexcept
    : data(other.data), backend(other.backend), size(other.size) {
    other.data = nullptr;
    other.backend = nullptr;
    other.size = CuboidShape{};
}

template<typename T>
ManagedData<T>& ManagedData<T>::operator=(ManagedData&& other) noexcept {
    if (this != &other) {
        // Free existing data if any
        if (data) {
            try {
                if (backend) {
                    backend->freeMemoryOnDevice(*this);
                }
            } catch(...){}
        }

        // Move data
        data = other.data;
        backend = other.backend;
        size = other.size;

        // Leave other in a valid state
        other.data = nullptr;
        other.backend = nullptr;
        other.size = CuboidShape{};
    }
    return *this;
}


