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

#include "ServiceAbstractions.h"
#include <memory>

// Forward declarations
class ThreadPool;
class Hyperstack;
class BaseDeconvolutionAlgorithm;
class DeconvolutionConfig;
class DeconvolutionStrategy;
class DeconvolutionAlgorithmFactory;

class DeconvolutionService : public IDeconvolutionService{
public:
    DeconvolutionService();
    ~DeconvolutionService() override;

    // IDeconvolutionService interface
    std::unique_ptr<DeconvolutionResult> deconvolve(const DeconvolutionRequest& request) override;

    virtual std::future<std::unique_ptr<DeconvolutionResult>> deconvolveAsync(const DeconvolutionRequest& request) override;
    
    virtual std::future<std::vector<std::unique_ptr<DeconvolutionResult>>> deconvolveBatchAsync(const std::vector<DeconvolutionRequest>& requests) override;
        

    virtual void setProgressCallback(std::function<void(int)> callback) override;
    std::vector<std::string> getSupportedAlgorithms() const override;
    std::vector<std::string> getSupportedStrategyTypes() const; 

    bool validateAlgorithmConfig(const std::string& algorithm, const json& config) const override;

    void setLogger(std::function<void(const std::string&)> logger) override;
    void setConfigLoader(std::function<json(const std::string&)> loader) override;
    std::shared_ptr<Hyperstack> loadImage(const std::string& path) override;

    // IService interface
    void initialize() override;
    bool isInitialized() const override;
    void shutdown() override;

    void setDefaultLogger(std::function<void(const std::string&)> logger) override;
    void setErrorHandler(std::function<void(const std::string&)> handler) override;

    // Additional methods for advanced deconvolution
    // std::unique_ptr<DeconvolutionResult> batchDeconvolute(
    //     const std::vector<DeconvolutionRequest>& requests);
    
    // std::unique_ptr<DeconvolutionResult> hybridDeconvolution(
    //     const DeconvolutionRequest& request,
    //     const std::vector<std::string>& algorithm_sequence);
    
    // std::vector<std::unique_ptr<DeconvolutionResult>> experimentDeconvolution(
    //     const DeconvolutionRequest& base_request,
    //     const std::vector<std::string>& algorithms_to_test);

private:
    void logMessage(const std::string& message);
    void handleError(const std::string& error);
    
    std::unique_ptr<DeconvolutionResult> createResult(
        bool success,
        const std::string& message,
        std::chrono::duration<double> duration);

    
    bool validateDeconvolutionRequest(const DeconvolutionRequest& request) const;
    bool validateImageConfig(const json& config) const;
    
    // Algorithm management
    std::unique_ptr<DeconvolutionStrategy> deconvolutionStrategy;

    // PSF package management
    std::vector<PSF> createPSFsFromSetup(
        std::shared_ptr<SetupConfig> setupConfig);


    // Configuration
    bool initialized_;
    std::function<void(const std::string&)> logger_;
    std::function<void(const std::string&)> error_handler_;
    std::function<json(const std::string&)> config_loader_;
    
    // Default handlers
    std::function<void(const std::string&)> default_logger_;
    std::function<void(const std::string&)> default_error_handler_;
    
    // Cached data
    std::vector<std::string> supported_algorithms_;

    // Multithreading
    std::unique_ptr<ThreadPool> thread_pool_;
    std::function<void(int)> progress_callback_;
    std::mutex progress_mutex_;
};