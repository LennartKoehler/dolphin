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
    std::cout << "=== Test: Shared Memory Tracking Across Backends ===" << std::endl;

    CPUBackendManager& mgr = getManager();

    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend1 = mgr.getBackend(config);
    IBackend& backend2 = mgr.getBackend(config);

    IBackendMemoryManager& memMgr1 = backend1.mutableMemoryManager();
    IBackendMemoryManager& memMgr2 = backend2.mutableMemoryManager();

    // CPU backends share the same MemoryTracking, so allocations on one
    // are visible in the other's allocated memory count.
    size_t baseline = memMgr1.getAllocatedMemory();

    CuboidShape shape(16, 16, 8);
    RealData data2 = memMgr2.allocateMemoryOnDeviceReal(shape);

    size_t afterAlloc = memMgr1.getAllocatedMemory();
    if (afterAlloc <= baseline) {
        std::cerr << "FAIL: Allocation on backend2 should be visible in backend1's memory tracking" << std::endl;
        return 1;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
