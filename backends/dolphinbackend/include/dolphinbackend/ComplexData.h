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
class IComputeBackend;

// Forward declare the TEMPLATE (not just a class!)
template<typename T>
class DataView;

typedef float real_t;
typedef real_t complex_t[2];

// A pass-by-value complex number.
//
// `complex_t` is a C-style array (real_t[2]), so it cannot be returned by value.
// Backend reductions (e.g. IComputeBackend::sum) return a ComplexValue instead,
// which is convertible to/from the raw complex_t array used across the backends.
struct ComplexValue {
    real_t re = 0;
    real_t im = 0;

    ComplexValue() = default;
    ComplexValue(real_t r, real_t i) : re(r), im(i) {}
    ComplexValue(complex_t v) : re(v[0]), im(v[1]) {}

    bool operator==(const ComplexValue& other) const { return re == other.re && im == other.im; }
};

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
    // Raw-data access is reserved for the backends. The interfaces are granted
    // friendship; their (and their derived implementations') members reach the
    // protected helpers below through normal inheritance.
    friend class IBackendMemoryManager;
    friend class IComputeBackend;

public:

    // Default constructor - initializes to empty state
    ManagedData() : data(nullptr), size{}, realSize{}, bytes(0), padding(0), backend(nullptr), ownsMemory(false) {}

    ManagedData(IBackendMemoryManager const* b, T* data, CuboidShape size, CuboidShape realSize, std::size_t bytes, std::size_t padding)
        : backend(b), data(data), size(size), realSize(realSize), bytes(bytes), padding(padding), ownsMemory(true) {}
    virtual ~ManagedData();

    ManagedData(const ManagedData& other);
    ManagedData& operator=(const ManagedData& other);

    ManagedData(ManagedData&& other) noexcept;
    ManagedData& operator=(ManagedData&& other) noexcept;

    class Iterator{
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = T;
        using pointer           = T*;  // or also value_type*
        using reference         = T&;  // or also value_type&

        Iterator(ManagedData<T>& owner, std::size_t index = 0, bool atEnd = false)
            : m_owner(&owner), m_index(index), m_atEnd(atEnd) {}

        reference operator*() const {
            return m_owner->access(m_index);
        }

        pointer operator->() {
            // For pointer semantics, return address of accessed element
            // Note: For strided data, this returns a temporary! Use with caution.
            return &m_owner->access(m_index);
        }

        Iterator& operator++() {
            ++m_index;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // Compare by owner + index (not raw pointer!)
        friend bool operator==(const Iterator& a, const Iterator& b) {
            return a.m_owner == b.m_owner && a.m_index == b.m_index && a.m_atEnd == b.m_atEnd;
        }

        friend bool operator!=(const Iterator& a, const Iterator& b) {
            return !(a == b);
        }

        // Convenience methods
        std::size_t getIndex() const { return m_index; }
        void setIndex(std::size_t idx) { m_index = idx; }

    private:
        ManagedData<T>* m_owner;
        std::size_t m_index;
        bool m_atEnd;
    };


    T& access(std::size_t linearIndex);
    const T& access(std::size_t linearIndex) const;

    // Raw buffer access. In production this is PROTECTED: only the backends can
    // reach it, via the friend interfaces (IBackendMemoryManager / IComputeBackend)
    // whose protected rawData() helper exposes it to their implementations.
    // Tests compile with DOLPHIN_TEST_RAW_ACCESS, which promotes the accessors to
    // public so they can inspect the buffer directly.
    //
    // WARNING (ODR): this macro must only be defined for test executables that
    // do NOT link the prebuilt dolphinbackend library (no explicit template
    // instantiations shared across macro-on/macro-off TUs). Defining it in a TU
    // that links the library creates two different class definitions of
    // ManagedData<T> (different access specifiers) and is a formal ODR
    // violation. Access specifiers do not affect layout or codegen, so this is
    // tolerated as an implementation detail for tests only.
#if defined(DOLPHIN_TEST_RAW_ACCESS)
public:
#else
protected:
#endif
    T* getData() const { return data; }
    void setData(T* p) { data = p; }

    // Claim ownership of the buffer this handle already points at (e.g. a DataView
    // that has provenance but does not free), so the destructor releases it.
    // Backend-internal: the validated public path is
    // IBackendMemoryManager::transferOwnership().
    void adoptMemory() { ownsMemory = true; }

    // Detach from the buffer without freeing: the destructor will not release it.
    // Used to hand ownership of the underlying memory to another handle (e.g. a view).
    // Backend-internal: the validated public path is
    // IBackendMemoryManager::transferOwnership().
    void releaseMemory() { data = nullptr; backend = nullptr; ownsMemory = false; }
#if !defined(DOLPHIN_TEST_RAW_ACCESS)
public:
#endif

    // Stride is computed as padding + width (memory row length)
    inline size_t getPadding() const { return padding; }
    inline void setPadding(size_t padding) { this->padding = padding; }

    size_t getDataBytes() const {return bytes;}
    size_t getValidBytes() const {return getSize().getVolume() * sizeof(T);}

    const CuboidShape& getRealSize() const { return realSize; }
    const CuboidShape& getSize() const { return size; }
    CuboidShape getPaddedSize() const {
        CuboidShape shape = realSize;
        shape.width += padding;
        return shape;
    }
    //WARNING getDataBytes doesnt have to be size.getVolume() * sizeof(T)!!!
    // for exmple the data might have padding at the end that has not data but is necessary for fft inplace
    virtual IBackendMemoryManager const* getBackend() const { return backend; }
    virtual void setBackend(IBackendMemoryManager const* backend) {this->backend = backend;}
    DataView<other_type_t<T>> reinterpret();

    // Check if data is valid
    virtual bool isValid() const { return data != nullptr && backend != nullptr; }
    explicit operator bool() const { return isValid(); }


    T& operator[](size_t index){return access(index);}
    const T& operator[](size_t index) const {return access(index);}
protected:
    size_t convertIndex(size_t linearIndex) const;

    T* data;
    CuboidShape size;
    CuboidShape realSize;
    std::size_t bytes;
    std::size_t padding; // number of elements after each row that are padding
    // data might be: p000,p001,p002,p003,pad,p010,p011,p012,p013,pad,padding
    // so this tells us how much padding there is after each row. getStride() returns padding + width
    IBackendMemoryManager const* backend;
    // Whether this handle owns (and will free) the buffer. Views (DataView) do not.
    bool ownsMemory;
};

// Version of ManagedData that can view and edit the data but never free it, used for complex -> real reinterpretation of the data
template<typename T>
class DataView : public ManagedData<T> {
public:
    // Non-owning empty view.
    DataView() :ManagedData<T>(){}

    // No copy — a shallow copy would alias the same buffer
    DataView(const DataView&) = delete;
    DataView& operator=(const DataView&) = delete;

    // Move delegates to ManagedData — properly nulls source
    DataView(DataView&& other) noexcept : ManagedData<T>(std::move(other)) {}
    DataView& operator=(DataView&& other) noexcept {
        ManagedData<T>::operator=(std::move(other));
        return *this;
    }

    // No-op: a view's provenance is fixed at construction (IBackendMemoryManager::
    // createDataView). Changing it afterwards would desynchronise isOnDevice from
    // the buffer's actual location.
    void setBackend(IBackendMemoryManager const*) override {}

    bool isValid() const override { return this->data != nullptr; }

    bool isView() const { return true; }

    ~DataView() = default;

private:
    // Views can only be created by the backends (via IBackendMemoryManager::
    // createDataView), which is the only place provenance is established. A view
    // aliases an existing buffer and never owns/frees it (ownsMemory == false).
    friend class IBackendMemoryManager;
    DataView(T* data, CuboidShape size, CuboidShape realSize, std::size_t bytes,
             std::size_t padding = 0)
        : ManagedData<T>() {
        this->data = data;
        this->size = size;
        this->realSize = realSize;
        this->bytes = bytes;
        this->padding = padding;
        this->ownsMemory = false;
    }
};


// Type aliases for backward compatibility
using ComplexData = ManagedData<complex_t>;

using RealData = ManagedData<real_t>;


using ComplexView = DataView<complex_t>;

using RealView = DataView<real_t>;

