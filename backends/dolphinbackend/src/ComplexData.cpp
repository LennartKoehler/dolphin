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
// Take ownership of pre-allocated memory

ComplexData::ComplexData(const IBackendMemoryManager* b, complex* data, RectangleShape size)
    : backend(b), data(data), size(size) {}

ComplexData::~ComplexData() {
    if (data) {
        try { backend->freeMemoryOnDevice(*this); } catch(...) {} // most likely the backend was deleted before the complexdata was freed
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
