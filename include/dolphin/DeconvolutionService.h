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

#include "dolphin/ProgressTracking.h"
#include "dolphin/ServiceAbstractions.h"
#include <memory>

// Forward declarations
class DeconvolutionStrategyPair;



class DeconvolutionService : public IService{
public:
    DeconvolutionService();
    ~DeconvolutionService() override;

    std::unique_ptr<DeconvolutionResult> deconvolve(const DeconvolutionRequest& request);



    std::vector<std::string> getSupportedAlgorithms() const;



    // IService interface
    void initialize() override;
    bool isInitialized() const override;
    void shutdown() override;
    void setLogger(std::shared_ptr<spdlog::logger> logger) override { logger_ = logger; }


private:
    std::unique_ptr<DeconvolutionResult> createResult(
        bool success,
        const std::string& message,
        std::chrono::duration<double> duration);

    void resolveMemory(SetupConfig& config) const;

    bool validateAlgorithmConfig(const std::string& algorithm) const;
    bool validateDeconvolutionRequest(const DeconvolutionRequest& request) const;

    // Algorithm management
    std::unique_ptr<DeconvolutionStrategyPair> deconvolutionStrategyPair;

    // Configuration
    bool initialized_;
    std::shared_ptr<spdlog::logger> logger_;
};
