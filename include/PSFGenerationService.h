#pragma once

#include "ServiceAbstractions.h"
#include "psf/configs/PSFConfig.h"
#include <memory>

class ThreadPool;
class PSFGeneratorFactory;
class BasePSFGenerator;

class PSFGenerationService : public IPSFGenerationService{
public:
    PSFGenerationService();
    ~PSFGenerationService() override;

    // IPSFGenerationService interface
    std::unique_ptr<PSFGenerationResult> generatePSF(const PSFGenerationRequest& request) override;
    std::future<std::unique_ptr<PSFGenerationResult>> generatePSFAsync(const PSFGenerationRequest& request) override;


    std::vector<std::string> getSupportedPSFTypes() const override;
    bool validateConfig(const json& config) const override;

    void setLogger(std::function<void(const std::string&)> logger) override;
    void setConfigLoader(std::function<json(const std::string&)> loader) override;

    // IService interface
    void initialize() override;
    bool isInitialized() const override;
    void shutdown() override;

    void setDefaultLogger(std::function<void(const std::string&)> logger) override;
    void setErrorHandler(std::function<void(const std::string&)> handler) override;


private:
    std::string savePSF(const std::string& path, const std::string& name, std::shared_ptr<PSF> psf);
    std::string savePSFConfig(const std::string& path, const std::string& name, std::shared_ptr<PSFConfig> psfconfig);
    std::string getExecutableDirectory();
    void logMessage(const std::string& message);
    void handleError(const std::string& error);
    
    std::unique_ptr<PSFGenerationResult> createResult(
        bool success,
        const std::string& message,
        std::chrono::duration<double> duration
    );
    
    std::unique_ptr<PSF> createPSFFromConfigInternal(std::shared_ptr<PSFConfig> psfConfig);
    std::unique_ptr<PSF> createPSFFromFilePathInternal(const std::string& path);
    
    bool isValidPSFType(const std::string& psf_type) const;
    
    // Dependencies
    PSFGeneratorFactory* generator_factory_;
    
    // Configuration
    bool initialized_;
    std::function<void(const std::string&)> logger_;
    std::function<void(const std::string&)> error_handler_;
    std::function<json(const std::string&)> config_loader_;
    
    // Default handlers
    std::function<void(const std::string&)> default_logger_;
    std::function<void(const std::string&)> default_error_handler_;
    
    // Cached data
    std::vector<std::string> supported_types_;
    std::string default_output_path_;

    std::unique_ptr<ThreadPool> thread_pool_;
    std::function<void(int)> progress_callback_;
    std::mutex progress_mutex_;
};