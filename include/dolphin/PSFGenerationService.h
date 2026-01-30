/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once

#include "dolphin/ServiceAbstractions.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include <memory>

class ThreadPool;
class PSFGeneratorFactory;
class BasePSFGenerator;

class PSFGenerationService : public IService{
public:
    PSFGenerationService();
    ~PSFGenerationService();

    // IPSFGenerationService interface
    std::unique_ptr<PSFGenerationResult> generatePSF(const PSFGenerationRequest& request);
    std::future<std::unique_ptr<PSFGenerationResult>> generatePSFAsync(const PSFGenerationRequest& request);


    std::vector<std::string> getSupportedPSFTypes() const;
    bool validateConfig(const json& config) const;


    // IService interface
    void initialize() override;
    bool isInitialized() const override;
    void shutdown() override;
    void setLogger(std::shared_ptr<spdlog::logger> logger) override { logger_ = logger; }


private:
    std::string savePSF(const std::string& path, const std::string& name, std::shared_ptr<PSF> psf);
    std::string savePSFConfig(const std::string& path, const std::string& name, std::shared_ptr<PSFConfig> psfconfig);
    std::string getExecutableDirectory();
    // void logMessage(const std::string& message);
    // void handleError(const std::string& error);
    
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
    std::shared_ptr<spdlog::logger> logger_;

    // Cached data
    std::vector<std::string> supported_types_;
    std::string default_output_path_;

    std::unique_ptr<ThreadPool> thread_pool_;
    std::function<void(int)> progress_callback_;
    std::mutex progress_mutex_;
};