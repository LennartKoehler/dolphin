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

// Forward declare the TEMPLATE (not just a class!)
template<typename T>
class DataView;

typedef float real_t;
typedef real_t complex_t[2];

// Type trait to compute the "other" data type (real_t <-> complex_t)
template<typename T>
struct other_type {
    using type = std::conditional_t<std::is_same_v<T, real_t>, complex_t, real_t>;
};

template<typename T>
using other_type_t = typename other_type<T>::type;


/**
 * @brief Templated RAII wrapper for managed memory buffers
 * @tparam T The data type (real_t for real-valued data, complex_t for complex-valued data)
 */
template<typename T>
class ManagedData {
public:
    using value_type = T;

    T* data;

    // Default constructor - initializes to empty state
    ManagedData() : data(nullptr), size{}, bytes(0), backend(nullptr) {}

    // Take ownership of pre-allocated memory
    ManagedData(IBackendMemoryManager const* b, T* data, CuboidShape size, std::size_t bytes);
    virtual ~ManagedData();

    ManagedData(const ManagedData& other);
    ManagedData& operator=(const ManagedData& other);

    ManagedData(ManagedData&& other) noexcept;
    ManagedData& operator=(ManagedData&& other) noexcept;

    void setData(void* data) {this->data = (T*)data;}

    // Accessors
    T* getData() { return data; }
    const T* getData() const { return data; }
    size_t getDataBytes() const {return bytes;}
    const CuboidShape& getSize() const { return size; }
    //WARNING getDataBytes doesnt have to be size.getVolume() * sizeof(T)!!!
    // for exmple the data might have padding at the end that has not data but is necessary for fft inplace
    virtual IBackendMemoryManager const* getBackend() const { return backend; }
    virtual void setBackend(IBackendMemoryManager const* backend) {this->backend = backend;}

    DataView<other_type_t<T>> reinterpret();


    // Check if data is valid
    virtual bool isValid() const { return data != nullptr && backend != nullptr; }
    explicit operator bool() const { return isValid(); }

protected:
    CuboidShape size;
    std::size_t bytes;
    IBackendMemoryManager const* backend;
};

// Version of ManagedData that can view and edit the data but never free it, used for complex -> real reinterpretation of the data
template<typename T>
class DataView : public ManagedData<T> {
public:
    // Constructor that accepts the same parameters but sets backend to nullptr
    // This ensures views don't own memory and the destructor won't try to free it
    DataView() :ManagedData<T>(){}
    DataView(IBackendMemoryManager const* /*b*/, T* data, CuboidShape size, std::size_t bytes) {
        this->data = data;
        this->size = size;
        this->bytes = bytes;
        this->backend = nullptr;  // Always nullptr - views don't own memory
    }

    DataView(const DataView& other) = default;
    DataView& operator=(const DataView& other) = default;

    DataView(DataView&& other) = default;
    DataView& operator=(DataView&& other) = default;
    // Override backend to always return nullptr (views don't own memory)
    IBackendMemoryManager const* getBackend() const override { return nullptr; }
    void setBackend(IBackendMemoryManager const* /*backend*/) override {}

    // isValid - only check data pointer, not backend
    bool isValid() const override { return this->data != nullptr; }

    // Tag method to identify views
    bool isView() const { return true; }

    // IMPORTANT: Default destructor - views don't own memory, so don't free it!
    ~DataView() = default;
};


// Type aliases for backward compatibility
using ComplexData = ManagedData<complex_t>;

using RealData = ManagedData<real_t>;


using ComplexView = DataView<complex_t>;

using RealView = DataView<real_t>;

