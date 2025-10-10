#include "backend/ComplexData.h"
#include "backend/IBackendMemoryManager.h"

// Take ownership of pre-allocated memory

ComplexData::ComplexData(IBackendMemoryManager* b, complex* data, RectangleShape size)
    : backend(b), data(data), size(size) {}

ComplexData::~ComplexData() {
    if (data) {
        try { backend->freeMemoryOnDevice(*this); } catch(...) {}
    }
}

