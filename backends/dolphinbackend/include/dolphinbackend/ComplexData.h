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
#include "CuboidShape.h"

class IBackendMemoryManager;

typedef float real_t;
typedef real_t complex_t[2];


/**
 * @brief Templated RAII wrapper for managed memory buffers
 * @tparam T The data type (real_t for real-valued data, complex_t for complex-valued data)
 */
template<typename T>
class ManagedData {
public:
    using value_type = T;

    T* data;
    CuboidShape size;
    const IBackendMemoryManager* backend;

    // Take ownership of pre-allocated memory
    ManagedData() = default;
    ManagedData(const IBackendMemoryManager* b, T* data, CuboidShape size);
    ~ManagedData();
    ManagedData(const ManagedData& other);
    ManagedData& operator=(const ManagedData& other);

    ManagedData(ManagedData&& other) noexcept;
    ManagedData& operator=(ManagedData&& other) noexcept;
    void setData(void* data) {this->data = (T*)data;}

    // Accessors
    T* getData() { return data; }
    const T* getData() const { return data; }
    size_t getDataBytes() const { return getElementSize() * getSize().getVolume();}
    const CuboidShape& getSize() const { return size; }
    size_t getElementSize() const {return sizeof(T);}
    const IBackendMemoryManager* getBackend() const { return backend; }

    // Check if data is valid
    bool isValid() const { return data != nullptr && backend != nullptr; }
    explicit operator bool() const { return isValid(); }
};


// Type aliases for backward compatibility
using ComplexData = ManagedData<complex_t>;

using RealData = ManagedData<real_t>;



