#include <iostream>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"

static CPUBackendManager& getManager() {
    static CPUBackendManager manager;
    static bool initialized = false;
    if (!initialized) {
        manager.init([](const std::string& msg, LogLevel level) {
            if (level >= LogLevel::ERROR) {
                std::cerr << "[CPU] " << msg << std::endl;
            }
        });
        initialized = true;
    }
    return manager;
}

int main() {
    std::cout << "=== Test: Multiple Independent Backends ===" << std::endl;

    CPUBackendManager& mgr = getManager();

    BackendConfig config1;
    config1.nThreads = 1;
    config1.backendName = "backend1";

    BackendConfig config2;
    config2.nThreads = 1;
    config2.backendName = "backend2";

    IBackend& backend1 = mgr.getBackend(config1);
    IBackend& backend2 = mgr.getBackend(config2);

    // Each backend should be a distinct object with its own deconv/memory
    if (&backend1 == &backend2) {
        std::cerr << "FAIL: Two backends should be distinct objects" << std::endl;
        return 1;
    }
    if (backend1.getDeviceString() != "cpu" || backend2.getDeviceString() != "cpu") {
        std::cerr << "FAIL: Both backends should report 'cpu'" << std::endl;
        return 1;
    }
    if (backend1.getMemoryManagerPtr() == backend2.getMemoryManagerPtr()) {
        std::cerr << "FAIL: Each backend should have its own memory manager" << std::endl;
        return 1;
    }
    if (!backend1.ownsDeconvolutionBackend() || !backend1.ownsMemoryManager()) {
        std::cerr << "FAIL: Backend1 should own its components" << std::endl;
        return 1;
    }
    if (!backend2.ownsDeconvolutionBackend() || !backend2.ownsMemoryManager()) {
        std::cerr << "FAIL: Backend2 should own its components" << std::endl;
        return 1;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
