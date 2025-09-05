#pragma once

#include "ServiceAbstractions.h"
#include "PSFGenerationService.h"
#include "DeconvolutionService.h"

class ServiceFactoryImpl : public ServiceFactory {
public:
    // ServiceFactory interface
    std::unique_ptr<IPSFGenerationService> createPSFGenerationService() override;
    std::unique_ptr<IDeconvolutionService> createDeconvolutionService() override;

    void setLogger(std::function<void(const std::string&)> logger) override;
    void setConfigLoader(std::function<json(const std::string&)> loader) override;

    // Singleton instance access
    static ServiceFactoryImpl& getInstance();

private:
    ServiceFactoryImpl() = default;
    ~ServiceFactoryImpl() = default;
    ServiceFactoryImpl(const ServiceFactoryImpl&) = delete;
    ServiceFactoryImpl& operator=(const ServiceFactoryImpl&) = delete;

    // Internal state
    std::function<void(const std::string&)> logger_;
    std::function<json(const std::string&)> config_loader_;
    bool logger_set_ = false;
    bool config_loader_set_ = false;
    
    // Friend function for singleton access
    friend std::unique_ptr<ServiceFactory> ServiceFactory::create();
};

// Custom deleter for singleton
struct ServiceFactoryDeleter {
    void operator()(ServiceFactoryImpl* p) const {
        // Don't delete singleton instance
    }
};

// Inline factory implementation for easy access
inline std::unique_ptr<ServiceFactory> ServiceFactory::create() {
    static ServiceFactoryDeleter deleter;
    return std::unique_ptr<ServiceFactory>(&ServiceFactoryImpl::getInstance(), deleter);
}