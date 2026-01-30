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

#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <functional>
#include "nlohmann/json.hpp"
#include "dolphin/frontend/SetupConfig.h"
#include <future>

#include <spdlog/spdlog.h>
class DeconvolutionService;
class PSFGenerationService;

using json = nlohmann::json;

// Forward declarations
class PSF;
class PSFConfig;

// --- Common Interface ---
class IService {
public:
    virtual ~IService() = default;
    virtual void initialize() = 0;
    virtual bool isInitialized() const = 0;
    virtual void shutdown() = 0;
    
    virtual void setLogger(std::shared_ptr<spdlog::logger> logger) = 0;
};

// --- Service Result Types ---
class ServiceResult {
public:
    virtual bool success() const = 0;
    virtual const std::string& errorMessage() const = 0;
    virtual const std::string& successMessage() const = 0;
    virtual std::chrono::duration<double> duration() const = 0;
    
    virtual ~ServiceResult() = default;
};

class ServiceResultBase : public ServiceResult {
protected:
    bool success_;
    std::string error_message_;
    std::string success_message_;
    std::chrono::duration<double> duration_;
    
public:
    ServiceResultBase(bool success, const std::string& message = "", 
                     std::chrono::duration<double> dur = std::chrono::duration<double>::zero())
        : success_(success), error_message_(message), success_message_(message), duration_(dur) {}
    
    bool success() const override { return success_; }
    const std::string& errorMessage() const override { return error_message_; }
    const std::string& successMessage() const override { return success_message_; }
    std::chrono::duration<double> duration() const override { return duration_; }
};

// --- PSF Service Abstractions ---
//basically wrapper for psfconfig
class PSFGenerationRequest {
public:
    struct PSFConfigInfo{
        std::string config_path_;
        std::shared_ptr<PSFConfig> psf_config_;
    };
    PSFGenerationRequest() = default;
    PSFGenerationRequest(std::shared_ptr<PSFConfig> config) {config_.psf_config_ = config;}
    PSFGenerationRequest(std::string path) { config_.config_path_ = path; }


    void setConfig(PSFConfigInfo config) { config_ = config; }
    PSFConfigInfo getConfig() const { return config_; }
 

    PSFConfigInfo config_;
    std::string output_path;
    bool save_result = false;
    bool show_example = false;
    // int num_threads = 1;

};

class PSFGenerationResult : public ServiceResultBase {
public:
    PSFGenerationResult(bool success, const std::string& message = "", 
                        std::chrono::duration<double> dur = std::chrono::duration<double>::zero())
        : ServiceResultBase(success, message, dur) {}
    
    std::shared_ptr<PSF> psf;
    std::string generated_path;
};

// class IPSFGenerationService : public IService{
// public:
//     virtual ~IPSFGenerationService() = default;
    
//     virtual std::unique_ptr<PSFGenerationResult> generatePSF(const PSFGenerationRequest& request) = 0;
//     virtual std::future<std::unique_ptr<PSFGenerationResult>> generatePSFAsync(const PSFGenerationRequest& request) = 0;
//     virtual std::vector<std::string> getSupportedPSFTypes() const = 0;
//     virtual bool validateConfig(const json& config) const = 0;
    
//     virtual void setLogger(std::function<void(const std::string&)> logger) = 0;
//     virtual void setConfigLoader(std::function<json(const std::string&)> loader) = 0;
// };

// --- Deconvolution Service Abstractions ---
class DeconvolutionRequest {
public:

    
    DeconvolutionRequest(std::shared_ptr<SetupConfig> config) : setup_config_(config) {
        output_path = config->outputDir;
    }
    
    void setConfig(std::shared_ptr<SetupConfig> config) { setup_config_ = config; }
    std::shared_ptr<SetupConfig> getConfig() const { return setup_config_; }

    void setPSFConfig(std::shared_ptr<PSFConfig> config) { psf_config_ = config; }
    std::shared_ptr<PSFConfig> getPSFConfig() const { return psf_config_; }

    bool save_separate = false;
    bool save_subimages = false;
    bool show_example = false;
    bool print_info = false;
    // int num_threads = 1;
    std::string output_path = "../results/deconv.tif";
    
    
private:
    std::shared_ptr<PSFConfig> psf_config_;
    std::shared_ptr<SetupConfig> setup_config_;

};

class DeconvolutionResult : public ServiceResultBase {
public:
    DeconvolutionResult(bool success, const std::string& message = "", 
                        std::chrono::duration<double> dur = std::chrono::duration<double>::zero())
        : ServiceResultBase(success, message, dur) {}
    
    std::string output_path;
    std::vector<std::string> individual_layer_paths;
    
    struct AlgorithmStats {
        std::string algorithm_used;
        std::chrono::duration<double> processing_time;
        double memory_usage_mb;
    };
    
    AlgorithmStats stats;
};

// class IDeconvolutionService : public IService {
// public:
//     virtual ~IDeconvolutionService() = default;
    
//     virtual std::unique_ptr<DeconvolutionResult> deconvolve(const DeconvolutionRequest& request) = 0;
//     // Asynchronous  
//     virtual std::future<std::unique_ptr<DeconvolutionResult>> deconvolveAsync(const DeconvolutionRequest& request) = 0;
    
//     // Batch processing
//     virtual std::future<std::vector<std::unique_ptr<DeconvolutionResult>>> deconvolveBatchAsync(
//         const std::vector<DeconvolutionRequest>& requests) = 0;
        
//     virtual void setProgressCallback(std::function<void(int)> callback) = 0;

//     // virtual std::unique_ptr<DeconvolutionResult> deconvolveFromConfig(const json& config) = 0;
//     virtual std::vector<std::string> getSupportedAlgorithms() const = 0;
//     virtual bool validateAlgorithmConfig(const std::string& algorithm, const json& config) const = 0;
    
//     virtual void setLogger(std::function<void(const std::string&)> logger) = 0;
//     virtual void setConfigLoader(std::function<json(const std::string&)> loader) = 0;
// };

// --- Service Factory ---
class ServiceFactory {
public:
    virtual ~ServiceFactory() = default;
    
    virtual std::unique_ptr<PSFGenerationService> createPSFGenerationService() = 0;
    virtual std::unique_ptr<DeconvolutionService> createDeconvolutionService() = 0;
    
    // virtual void setLogger(std::function<void(const std::string&)> logger) = 0;
    // virtual void setConfigLoader(std::function<json(const std::string&)> loader) = 0;
    
    static ServiceFactory* create();
};

