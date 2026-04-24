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
#include <memory>

// Forward declarations
class ThreadPool;
class Hyperstack;
class BaseDeconvolutionAlgorithm;
class DeconvolutionConfig;
class DeconvolutionStrategy;
class DeconvolutionStrategyPair;
class DeconvolutionAlgorithmFactory;

class DeconvolutionService : public IService{
public:
    DeconvolutionService();
    ~DeconvolutionService() override;

    // IDeconvolutionService interface
    std::unique_ptr<DeconvolutionResult> deconvolve(const DeconvolutionRequest& request);

    virtual std::future<std::unique_ptr<DeconvolutionResult>> deconvolveAsync(const DeconvolutionRequest& request);



    std::vector<std::string> getSupportedAlgorithms() const;
    std::vector<std::string> getSupportedStrategyTypes() const;



    // IService interface
    void initialize() override;
    bool isInitialized() const override;
    void shutdown() override;
    void setLogger(std::shared_ptr<spdlog::logger> logger) override { logger_ = logger; }


private:
    // void logMessage(const std::string& message);
    // void handleError(const std::string& error);

    std::unique_ptr<DeconvolutionResult> createResult(
        bool success,
        const std::string& message,
        std::chrono::duration<double> duration);


    bool validateAlgorithmConfig(const std::string& algorithm) const;
    bool validateDeconvolutionRequest(const DeconvolutionRequest& request) const;
    // bool validateImageConfig(const json& config) const;

    // Algorithm management
    std::unique_ptr<DeconvolutionStrategyPair> deconvolutionStrategyPair;

    // PSF package management
    std::vector<PSF> createPSFsFromSetup(
        std::shared_ptr<SetupConfig> setupConfig,
        const CuboidShape& imageShape);


    // Configuration
    bool initialized_;
    std::shared_ptr<spdlog::logger> logger_;

    // Multithreading
    std::unique_ptr<ThreadPool> thread_pool_;
};
