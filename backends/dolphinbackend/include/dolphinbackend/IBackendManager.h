#pragma once

#include "IBackendMemoryManager.h"
#include "IDeconvolutionBackend.h"
#include "IBackend.h"


enum LogLevel { DEBUG = 0, INFO, WARN, ERROR };
using LogCallback = std::function<void(const std::string& message, LogLevel level)>;

struct BackendConfig{
    BackendConfig(std::string backendName, int nThreads, LogCallback fn) : backendName(backendName), nThreads(nThreads) {}
    BackendConfig(std::string backendName, int nThreads) : backendName(backendName), nThreads(nThreads){}
 
    std::string backendName = "default";
    int nThreads;
    std::string deviceID;
};

class IBackendManager{
public: 
    IBackendManager() = default;
    virtual ~IBackendManager() = default;


    virtual void setLogger(LogCallback fn) = 0;

    virtual IDeconvolutionBackend& getDeconvolutionBackend(const BackendConfig& config) = 0;

    virtual IBackendMemoryManager& getBackendMemoryManager(const BackendConfig& config) = 0;
    
    virtual IBackend& getBackend(const BackendConfig& config) = 0;

};