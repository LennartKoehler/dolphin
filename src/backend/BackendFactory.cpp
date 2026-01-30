#include "dolphin/backend/BackendFactory.h"
#include "cpu_backend/CPUBackend.h"

CPUBackendMemoryManager& BackendFactory::getDefaultBackendMemoryManager() {
    static CPUBackendMemoryManager mgr;
    static bool init = (mgr.setMemoryLimit(CPUBackendMemoryManager::staticGetAvailableMemory()), true);

    static bool setlogger = (set_backend_logger(logCallback_fn), true);
    (void)init; // silence unused warning
    return mgr;
}