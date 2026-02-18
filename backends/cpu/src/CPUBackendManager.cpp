#include "CPUBackendManager.h"
#include "CPUBackend.h"



void CPUBackendManager::setLogger(LogCallback fn) {
	std::lock_guard<std::mutex> lock(mutex_);
	logger_ = std::move(fn);
}

IDeconvolutionBackend& CPUBackendManager::getDeconvolutionBackend(const BackendConfig& config) {
	BackendConfig localConfig = config;
	auto deconv = std::make_unique<CPUDeconvolutionBackend>(*this);
	deconv->init(localConfig);

    deconvBackends.push_back(std::move(deconv));
	return *deconvBackends.back();
}

IBackendMemoryManager& CPUBackendManager::getBackendMemoryManager(const BackendConfig& config) {
	BackendConfig localConfig = config;
	auto manager = std::make_unique<CPUBackendMemoryManager>(*this);
	manager->init(localConfig);

    memoryManagers.push_back(std::move(manager));
	return *memoryManagers.back();
}

IBackend& CPUBackendManager::getBackend(const BackendConfig& config) {
	BackendConfig localConfig = config;
	auto* backend = CPUBackend::create(*this);
	backend->init(localConfig);
    backends.push_back(std::unique_ptr<CPUBackend>(backend));
    
	return *backend;
}

