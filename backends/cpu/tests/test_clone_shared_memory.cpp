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
    std::cout << "=== Test: Clone Shared Memory Backend ===" << std::endl;

    CPUBackendManager& mgr = getManager();

    BackendConfig config;
    config.nThreads = 1;

    IBackend& original = mgr.getBackend(config);
    IBackend& cloned = mgr.cloneSharedMemory(original, config);

    // Cloned should be a separate backend object
    if (&original == &cloned) {
        std::cerr << "FAIL: Cloned backend should be a different object" << std::endl;
        return 1;
    }

    if (cloned.getDeviceString() != "cpu") {
        std::cerr << "FAIL: Cloned backend should report 'cpu'" << std::endl;
        return 1;
    }

    // Cloned should own its own components (it's a fresh backend from getBackend)
    if (!cloned.ownsDeconvolutionBackend()) {
        std::cerr << "FAIL: Cloned should own its deconv" << std::endl;
        return 1;
    }
    if (!cloned.ownsMemoryManager()) {
        std::cerr << "FAIL: Cloned should own its memory manager" << std::endl;
        return 1;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
