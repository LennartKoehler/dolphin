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
    std::cout << "=== Test: Move Data Between Backends ===" << std::endl;

    CPUBackendManager& mgr = getManager();

    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend1 = mgr.getBackend(config);
    IBackend& backend2 = mgr.getBackend(config);

    IBackendMemoryManager& memMgr1 = backend1.mutableMemoryManager();
    IBackendMemoryManager& memMgr2 = backend2.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);

    // Allocate and fill on backend1
    RealData src = memMgr1.allocateMemoryOnDeviceReal(shape);
    for (int i = 0; i < shape.getVolume(); ++i) {
        src[i] = static_cast<real_t>(i * 5);
    }

    // Move data from backend1 to backend2
    void* movedPtr = memMgr1.moveDataFromDevice(src.getData(), bytes, shape, memMgr2);
    if (movedPtr == nullptr) {
        std::cerr << "FAIL: Moved data pointer should not be null" << std::endl;
        return 1;
    }

    // For CPU backend moving to CPU backend, moveDataFromDevice returns the same pointer
    if (movedPtr != src.getData()) {
        std::cerr << "FAIL: CPU-to-CPU move should return same pointer" << std::endl;
        return 1;
    }

    // Don't double-free: since moveDataFromDevice returned the same pointer,
    // src still owns it and will free it on destruction. Unlink src from it.
    src.setData(nullptr);
    memMgr2.freeMemoryOnDevice(movedPtr, bytes);

    std::cout << "PASSED" << std::endl;
    return 0;
}
