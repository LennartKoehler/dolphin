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
#include "dolphin/Logging.h"
#include "nlohmann/json.hpp"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/ProgressTracking.h"
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


template<typename T>
struct Result {

    Result() = default;
    template<typename U>
    Result(U&& value, std::string error, bool success)
    : value(std::forward<U>(value)),
    success(success){
        if (!success) errors.push_back(error);
    }



    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    Result(Result&&) = default;
    Result& operator=(Result&&) = default;


    static Result<T> ok(T&& v){
        Result<T> r;
        r.value = std::move(v);
        r.success = true;
        return r;
    }

    static Result<T> fail(std::string e){
        Result<T> r;
        r.success = false;
        r.errors.push_back(std::move(e));
        return r;
    }

    template<typename otherT>
    Result(const Result<otherT>& other){
        if( !other.success){
            success = false;
            errors.insert(errors.end(),
                          other.errors.begin(),
                          other.errors.end());
        }
    }

    std::string getErrorString() const {
        std::string s;
        for (std::string error : errors){
            s += error;
        }
        return s;
    }

    T value;
    std::vector<std::string> errors;
    bool success{true};
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
    PSFGenerationRequest() = default;
    PSFGenerationRequest(std::shared_ptr<SetupConfigPSF> setupConfig) { this->setup_config_ = setupConfig; }
    PSFGenerationRequest(std::shared_ptr<SetupConfigPSF> setupConfig,
                         Logging::LogCallback frontendLogging,
                         progressCallbackFn progressCallback)
        : setup_config_(setupConfig), frontendLogging(frontendLogging), progressCallback(progressCallback) {}

    void setConfig(std::shared_ptr<SetupConfigPSF> config) { setup_config_ = config; }
    std::shared_ptr<SetupConfigPSF> getConfig() const { return setup_config_; }

    void setInlinePSFConfigs(std::vector<std::shared_ptr<PSFConfig>> configs) { inline_psf_configs_ = std::move(configs); }
    const std::vector<std::shared_ptr<PSFConfig>>& getInlinePSFConfigs() const { return inline_psf_configs_; }
    bool hasInlinePSFConfigs() const { return !inline_psf_configs_.empty(); }

    void setProgressCallback(progressCallbackFn fn) {this->progressCallback = fn;}
    progressCallbackFn getProgressCallback() const {return progressCallback;}

    void setFrontendLogging(Logging::LogCallback fl) {this->frontendLogging = fl;}
    Logging::LogCallback getFrontendLogging() const {return frontendLogging;}

private:

    std::shared_ptr<SetupConfigPSF> setup_config_;
    std::vector<std::shared_ptr<PSFConfig>> inline_psf_configs_;
    Logging::LogCallback frontendLogging;
    progressCallbackFn progressCallback;
};

class PSFGenerationResult : public ServiceResultBase {
public:
    PSFGenerationResult(bool success, const std::string& message = "",
                        std::chrono::duration<double> dur = std::chrono::duration<double>::zero())
        : ServiceResultBase(success, message, dur) {}

    std::shared_ptr<PSF> psf;
    std::string generated_path;
};

// --- Deconvolution Service Abstractions ---
class DeconvolutionRequest {
public:
    DeconvolutionRequest() = default;
    DeconvolutionRequest(std::shared_ptr<SetupConfig> config, std::shared_ptr<DeconvolutionConfig> deconvConfig)
        : setup_config_(config), deconv_config_(deconvConfig) {}
    DeconvolutionRequest(std::shared_ptr<SetupConfig> setupConfig,
                         std::shared_ptr<DeconvolutionConfig> deconvConfig,
                         progressCallbackFn progressCallback)
        : progressCallback(progressCallback),
          setup_config_(setupConfig), deconv_config_(deconvConfig) {}
    DeconvolutionRequest(std::shared_ptr<SetupConfig> setupConfig,
                         std::shared_ptr<DeconvolutionConfig> deconvConfig,
                         Logging::LogCallback frontendLogging,
                         progressCallbackFn progressCallback)
        : progressCallback(progressCallback), frontendLogging(frontendLogging),
          setup_config_(setupConfig), deconv_config_(deconvConfig) {}

    void setConfig(std::shared_ptr<SetupConfig> config) { setup_config_ = config; }
    std::shared_ptr<SetupConfig> getConfig() const { return setup_config_; }

    void setDeconvolutionConfig(std::shared_ptr<DeconvolutionConfig> config) { deconv_config_ = config; }
    std::shared_ptr<DeconvolutionConfig> getDeconvolutionConfig() const { return deconv_config_; }

    void setInlinePSFConfigs(std::vector<std::shared_ptr<PSFConfig>> configs) { inline_psf_configs_ = std::move(configs); }
    const std::vector<std::shared_ptr<PSFConfig>>& getInlinePSFConfigs() const { return inline_psf_configs_; }
    bool hasInlinePSFConfigs() const { return !inline_psf_configs_.empty(); }

    void setProgressCallback(progressCallbackFn fn) {this->progressCallback = fn;}
    progressCallbackFn getProgressCallback() const {return progressCallback;}

    void setFrontendLogging(Logging::LogCallback fl) {this->frontendLogging = fl;}
    Logging::LogCallback getFrontendLogging() const {return frontendLogging;}


private:
    progressCallbackFn progressCallback;
    Logging::LogCallback frontendLogging;
    std::vector<std::shared_ptr<PSFConfig>> inline_psf_configs_;
    std::shared_ptr<SetupConfig> setup_config_;
    std::shared_ptr<DeconvolutionConfig> deconv_config_;

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

// --- Service Factory ---
class ServiceFactory {
public:
    virtual ~ServiceFactory() = default;

    virtual std::unique_ptr<PSFGenerationService> createPSFGenerationService() = 0;
    virtual std::unique_ptr<DeconvolutionService> createDeconvolutionService() = 0;

    static ServiceFactory* create();
};

