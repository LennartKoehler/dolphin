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
#include <format>

namespace dolphin {
namespace backend {

// Base exception class for all backend-related exceptions
class BackendException : public std::runtime_error {
public:
    explicit BackendException(const std::string& message, 
                            const std::string& backend_type = "unknown",
                            const std::string& operation = "unknown")
        : std::runtime_error(message), 
          backend_type_(backend_type), 
          operation_(operation) {}

    const std::string& getBackendType() const { return backend_type_; }
    const std::string& getOperation() const { return operation_; }

    virtual std::string getDetailedMessage() const {
        return std::string(what()) + 
               " [Backend: " + backend_type_ + 
               ", Operation: " + operation_ + "]";
    }

protected:
    std::string backend_type_;
    std::string operation_;
};

// Specialized exception for memory-related errors
class MemoryException : public BackendException {
public:
    explicit MemoryException(const std::string& message, 
                           const std::string& backend_type = "unknown",
                           size_t requested_size = 0,
                           const std::string& operation = "unknown")
        : BackendException(message, backend_type, operation), 
          requested_size_(requested_size) {}

    size_t getRequestedSize() const { return requested_size_; }

    std::string getDetailedMessage() const override {
        return std::string(what()) + 
               " [Backend: " + backend_type_ + 
               ", Operation: " + operation_ + 
               (requested_size_ > 0 ? std::format(", Requested Size: {:.2f} GB", (static_cast<int>(requested_size_) / 1e9)) : "") + "]";
    }

private:
    size_t requested_size_;
};

} // namespace backend
} // namespace dolphin

// Unified memory allocation check macro
#define MEMORY_ALLOC_CHECK(ptr, size, backend_type, operation) { \
    if (ptr == nullptr) { \
        throw dolphin::backend::MemoryException( \
            "Memory allocation failed", \
            backend_type, \
            size, \
            operation \
        ); \
    } \
}



// General backend error check macro
#define BACKEND_CHECK(condition, message, backend_type, operation) { \
    if (!(condition)) { \
        throw dolphin::backend::BackendException( \
            message, \
            backend_type, \
            operation \
        ); \
    } \
}
