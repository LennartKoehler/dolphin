#include "backend/ComplexData.h"
#include "backend/IBackendMemoryManager.h"

// Take ownership of pre-allocated memory

ComplexData::ComplexData(const IBackendMemoryManager* b, complex* data, RectangleShape size)
    : backend(b), data(data), size(size) {}

ComplexData::~ComplexData() {
    if (data) {
        try { backend->freeMemoryOnDevice(*this); } catch(...) {}
    }
}


ComplexData::ComplexData(const ComplexData& other)
    : backend(other.backend), size(other.size) {
    ComplexData copy = backend->copyData(other);
    this->data = copy.data;
    copy.data = nullptr; // Prevent copy's destructor from freeing the data
}

ComplexData& ComplexData::operator=(const ComplexData& other){
    if (this != &other) {
        // Free existing data
        if (data) {
            try { backend->freeMemoryOnDevice(*this); } catch(...) {}
        }
        
        // Copy from other
        backend = other.backend;
        size = other.size;
        ComplexData copy = backend->copyData(other);
        data = copy.data;
        copy.data = nullptr; // Prevent copy's destructor from freeing the data
    }
    return *this;
}

ComplexData::ComplexData(ComplexData&& other) noexcept 
    : data(other.data), backend(other.backend), size(other.size) {
    other.data = nullptr;
}