#pragma once

#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <functional>
#include "../lib/nlohmann/json.hpp"
#include "HyperstackImage.h"

using json = nlohmann::json;

// Forward declarations
class SetupConfig;
class PSF;
class HyperstackImage;
class PSFConfig;

// --- Common Interface ---
class IService {
public:
    virtual ~IService() = default;
    virtual void initialize() = 0;
    virtual bool isInitialized() const = 0;
    virtual void shutdown() = 0;
    
    virtual void setDefaultLogger(std::function<void(const std::string&)> logger) = 0;
    virtual void setErrorHandler(std::function<void(const std::string&)> handler) = 0;
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
};

class PSFGenerationResult : public ServiceResultBase {
public:
    PSFGenerationResult(bool success, const std::string& message = "", 
                        std::chrono::duration<double> dur = std::chrono::duration<double>::zero())
        : ServiceResultBase(success, message, dur) {}
    
    std::shared_ptr<PSF> psf;
    std::string generated_path;
};

class IPSFGenerationService : public IService{
public:
    virtual ~IPSFGenerationService() = default;
    
    virtual std::unique_ptr<PSFGenerationResult> generatePSF(const PSFGenerationRequest& request) = 0;
    virtual std::unique_ptr<PSFGenerationResult> generatePSFFromConfig(std::shared_ptr<PSFConfig> config) = 0;
    virtual std::unique_ptr<PSFGenerationResult> generatePSFFromFilePath(const std::string& path) = 0;
    virtual std::vector<std::string> getSupportedPSFTypes() const = 0;
    virtual bool validateConfig(const json& config) const = 0;
    
    virtual void setLogger(std::function<void(const std::string&)> logger) = 0;
    virtual void setConfigLoader(std::function<json(const std::string&)> loader) = 0;
};

// --- Deconvolution Service Abstractions ---
class DeconvolutionRequest {
public:

    
    DeconvolutionRequest(std::shared_ptr<SetupConfig> config) : setup_config_(config) {}
    
    void setConfig(std::shared_ptr<SetupConfig> config) { setup_config_ = config; }
    std::shared_ptr<SetupConfig> getConfig() const { return setup_config_; }

    void setPSFConfig(std::shared_ptr<PSFConfig> config) { psf_config_ = config; }
    std::shared_ptr<PSFConfig> getPSFConfig() const { return psf_config_; }


    bool save_separate = false;
    bool save_subimages = false;
    bool show_example = false;
    bool print_info = false;
    std::string output_path = "../result/deconv.tif";
    
    
private:
    std::shared_ptr<PSFConfig> psf_config_;
    std::shared_ptr<SetupConfig> setup_config_;

};

class DeconvolutionResult : public ServiceResultBase {
public:
    DeconvolutionResult(bool success, const std::string& message = "", 
                        std::chrono::duration<double> dur = std::chrono::duration<double>::zero())
        : ServiceResultBase(success, message, dur) {}
    
    std::shared_ptr<Hyperstack> result;
    std::string output_path;
    std::vector<std::string> individual_layer_paths;
    
    struct AlgorithmStats {
        std::string algorithm_used;
        std::chrono::duration<double> processing_time;
        double memory_usage_mb;
    };
    
    AlgorithmStats stats;
};

class IDeconvolutionService : public IService {
public:
    virtual ~IDeconvolutionService() = default;
    
    virtual std::unique_ptr<DeconvolutionResult> deconvolve(const DeconvolutionRequest& request) = 0;
    virtual std::unique_ptr<DeconvolutionResult> deconvolveFromConfig(const json& config) = 0;
    virtual std::vector<std::string> getSupportedAlgorithms() const = 0;
    virtual bool validateAlgorithmConfig(const std::string& algorithm, const json& config) const = 0;
    
    virtual void setLogger(std::function<void(const std::string&)> logger) = 0;
    virtual void setConfigLoader(std::function<json(const std::string&)> loader) = 0;
    virtual std::shared_ptr<Hyperstack> loadImage(const std::string& path) = 0;
};

// --- Service Factory ---
class ServiceFactory {
public:
    virtual ~ServiceFactory() = default;
    
    virtual std::unique_ptr<IPSFGenerationService> createPSFGenerationService() = 0;
    virtual std::unique_ptr<IDeconvolutionService> createDeconvolutionService() = 0;
    
    virtual void setLogger(std::function<void(const std::string&)> logger) = 0;
    virtual void setConfigLoader(std::function<json(const std::string&)> loader) = 0;
    
    static std::unique_ptr<ServiceFactory> create();
};

