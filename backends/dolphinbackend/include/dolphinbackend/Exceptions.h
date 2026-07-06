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
#include <cstdio>
#include <cassert>
#include <sstream>
#include <thread>

namespace dolphin {
namespace backend {

inline std::string buildCpuContext() {
    std::ostringstream ctx;
    ctx << "cpu:cpu:tid:" << std::this_thread::get_id();
    return ctx.str();
}

// Base exception class for all backend-related exceptions
class BackendException : public std::runtime_error {
public:
    explicit BackendException(const std::string& message,
                            const std::string& backend_type = "unknown",
                            const std::string& operation = "unknown",
                            const std::string& context = "")
        : std::runtime_error(message),
          backend_type_(backend_type),
          operation_(operation),
          context_(context) {}

    const std::string& getBackendType() const { return backend_type_; }
    const std::string& getOperation() const { return operation_; }
    const std::string& getContext() const { return context_; }

    virtual std::string getDetailedMessage() const {
        std::string msg = std::string(what()) +
               " [Backend: " + backend_type_ +
               ", Operation: " + operation_;
        if (!context_.empty()) {
            msg += ", Context: " + context_;
        }
        msg += "]";
        return msg;
    }

protected:
    std::string backend_type_;
    std::string operation_;
    std::string context_;
};

// Specialized exception for memory-related errors
class MemoryException : public BackendException {
public:
    explicit MemoryException(const std::string& message,
                           const std::string& backend_type = "unknown",
                           size_t requested_size = 0,
                           const std::string& operation = "unknown",
                           const std::string& context = "")
        : BackendException(message, backend_type, operation, context),
          requested_size_(requested_size) {}

    size_t getRequestedSize() const { return requested_size_; }

    std::string getDetailedMessage() const override {
        std::string msg = std::string(what()) +
               " [Backend: " + backend_type_ +
               ", Operation: " + operation_;
        if (requested_size_ > 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), ", Requested Size: %.2f GB", static_cast<double>(requested_size_) / 1e9);
            msg += buf;
        }
        if (!context_.empty()) {
            msg += ", Context: " + context_;
        }
        msg += "]";
        return msg;
    }

private:
    size_t requested_size_;
};

} // namespace backend
} // namespace dolphin

// Unified memory allocation check macro
#define MEMORY_ALLOC_CHECK(ptr, size, backend_type, operation, ...) { \
    if (ptr == nullptr) { \
        throw dolphin::backend::MemoryException( \
            "Memory allocation failed", \
            backend_type, \
            size, \
            operation \
            __VA_OPT__(, ) __VA_ARGS__ \
        ); \
    } \
}



// General backend error check macro
#define BACKEND_CHECK(condition, message, backend_type, operation, ...) { \
    if (!(condition)) { \
        throw dolphin::backend::BackendException( \
            message, \
            backend_type, \
            operation \
            __VA_OPT__(, ) __VA_ARGS__ \
        ); \
    } }
