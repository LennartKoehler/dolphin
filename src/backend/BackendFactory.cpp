#include "dolphin/backend/BackendFactory.h"
#include "cpu_backend/CPUBackend.h"

CPUBackendMemoryManager& BackendFactory::getDefaultBackendMemoryManager() {
    static CPUBackendMemoryManager mgr;
    static bool init = (mgr.setMemoryLimit(CPUBackendMemoryManager::staticGetAvailableMemory()), true);

    static BackendConfig config{
        "default",
        1,
        logCallback_fn
    };
    static bool init2 = (mgr.init(config), true);
    (void)init; // silence unused warning
    return mgr;
}