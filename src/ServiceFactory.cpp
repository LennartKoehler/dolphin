#include "ServiceFactory.h"
#include "PSFGenerationService.h"
#include "DeconvolutionService.h"

// Thread-local singleton instance using Meyer's pattern
thread_local static ServiceFactoryImpl* tls_instance = nullptr;

std::unique_ptr<IPSFGenerationService> ServiceFactoryImpl::createPSFGenerationService() {
    auto service = std::make_unique<PSFGenerationService>();
    
    // Set up dependencies if they were pre-configured
    if (logger_set_) {
        service->setLogger(logger_);
    }
    
    if (config_loader_set_) {
        service->setConfigLoader(config_loader_);
    }
    
    return service;
}

std::unique_ptr<IDeconvolutionService> ServiceFactoryImpl::createDeconvolutionService() {
    auto service = std::make_unique<DeconvolutionService>();
    
    // Set up dependencies if they were pre-configured
    if (logger_set_) {
        service->setLogger(logger_);
    }
    
    if (config_loader_set_) {
        service->setConfigLoader(config_loader_);
    }
    
    return service;
}

void ServiceFactoryImpl::setLogger(std::function<void(const std::string&)> logger) {
    logger_ = logger;
    logger_set_ = true;
}

void ServiceFactoryImpl::setConfigLoader(std::function<json(const std::string&)> loader) {
    config_loader_ = loader;
    config_loader_set_ = true;
}

ServiceFactoryImpl& ServiceFactoryImpl::getInstance() {
    if (!tls_instance) {
        tls_instance = new ServiceFactoryImpl();
    }
    return *tls_instance;
}

// Implementation of static method declared in ServiceAbstractions.h
// std::unique_ptr<ServiceFactory> ServiceFactory::create() {
//     return std::make_unique<ServiceFactoryImpl>();
// }