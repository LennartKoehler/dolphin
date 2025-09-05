#pragma once

#include "ServiceAbstractions.h"
#include <memory>

// Forward declarations
class Hyperstack;
class BaseDeconvolutionAlgorithm;
class DeconvolutionConfig;
class DeconvolutionAlgorithmFactory;

class DeconvolutionService : public IDeconvolutionService{
public:
    DeconvolutionService();
    ~DeconvolutionService() override;

    // IDeconvolutionService interface
    std::unique_ptr<DeconvolutionResult> deconvolve(const DeconvolutionRequest& request) override;
    std::unique_ptr<DeconvolutionResult> deconvolveFromConfig(const json& config) override;
    std::vector<std::string> getSupportedAlgorithms() const override;
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
    std::unique_ptr<DeconvolutionResult> batchDeconvolute(
        const std::vector<DeconvolutionRequest>& requests);
    
    std::unique_ptr<DeconvolutionResult> hybridDeconvolution(
        const DeconvolutionRequest& request,
        const std::vector<std::string>& algorithm_sequence);
    
    std::vector<std::unique_ptr<DeconvolutionResult>> experimentDeconvolution(
        const DeconvolutionRequest& base_request,
        const std::vector<std::string>& algorithms_to_test);

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
    std::shared_ptr<BaseDeconvolutionAlgorithm> createAlgorithm(
        std::shared_ptr<SetupConfig> c1,
        std::shared_ptr<DeconvolutionConfig> c2);
    
    // PSF package management
    std::vector<PSF> createPSFsFromSetup(
        std::shared_ptr<SetupConfig> setupConfig);

    // Dependencies
    DeconvolutionAlgorithmFactory* algorithm_factory_;
    
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
};