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


    // Default constructor - initializes to empty state
    ManagedData() : data(nullptr), size{}, realSize{}, bytes(0), padding(0), backend(nullptr) {}

    ManagedData(IBackendMemoryManager const* b, T* data, CuboidShape size, CuboidShape realSize, std::size_t bytes, std::size_t padding);
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
    T& getValue(int x, int y, int z) ;//TODO

    inline T* getData() const { return data; }
    void setData(T* data) {this->data = data;}

    // Stride is computed as padding + width (memory row length)
    // inline size_t getStride() const { return size.width + padding; }
    // void setStride(size_t stride) {
    //     this->padding = stride - size.width;
    // }

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
    // data might be: p000,p001,p002,p003,pad,pad,p010,p011,p012,p013,pad,padding
    // so this tells us how much padding there is after each row. getStride() returns padding + width
    IBackendMemoryManager const* backend;
};

// Version of ManagedData that can view and edit the data but never free it, used for complex -> real reinterpretation of the data
template<typename T>
class DataView : public ManagedData<T> {
public:
    // Constructor that accepts the same parameters but sets backend to nullptr
    // This ensures views don't own memory and the destructor won't try to free it
    DataView() :ManagedData<T>(){}
    DataView(IBackendMemoryManager const* /*b*/, T* data, CuboidShape size, CuboidShape realSize, std::size_t bytes,
             std::size_t padding = 0) {
        this->data = data;
        this->size = size;
        this->realSize = realSize;
        this->bytes = bytes;
        this->backend = nullptr;  // Always nullptr - views don't own memory
        this->padding = padding;
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

