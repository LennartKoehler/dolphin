#pragma once

#include "dolphinbackend/IBackendManager.h"
#include <mutex>

class CPUBackend;
class CPUBackendMemoryManager;
class CPUDeconvolutionBackend;

class CPUBackendManager : public IBackendManager{
public: 

    CPUBackendManager() = default;
    ~CPUBackendManager() override = default;
    void setLogger(LogCallback fn) override;

    IDeconvolutionBackend& getDeconvolutionBackend(const BackendConfig& config) override;
    IBackendMemoryManager& getBackendMemoryManager(const BackendConfig& config) override;
    IBackend& getBackend(const BackendConfig& config) override;

    CPUBackendManager(const CPUBackendManager&) = delete;
    CPUBackendManager& operator=(const CPUBackendManager&) = delete;

private:

    std::vector<std::unique_ptr<CPUBackend>> backends;
    std::vector<std::unique_ptr<CPUDeconvolutionBackend>> deconvBackends;
    std::vector<std::unique_ptr<CPUBackendMemoryManager>> memoryManagers;

    LogCallback logger_;
    std::mutex mutex_;
};
