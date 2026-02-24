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
#include <vector>
#include <memory>

#include "dolphin/ServiceAbstractions.h"
#include "dolphin/ServiceFactory.h"
#include <unordered_map>

using json = nlohmann::json;

class Dolphin{
public:
    Dolphin() = default;
    ~Dolphin(){}

    void init();

    std::unique_ptr<PSFGenerationResult> generatePSF(PSFGenerationRequest request); // should prob just take the request
    std::unique_ptr<DeconvolutionResult> deconvolve(DeconvolutionRequest request);

    std::future<std::unique_ptr<PSFGenerationResult>> generatePSFAsync(PSFGenerationRequest request);
    std::future<std::unique_ptr<DeconvolutionResult>> deconvolveAsync(DeconvolutionRequest request);


private:
 
    // Service layer components (abstracted)
    ServiceFactory* service_factory_;
    std::unique_ptr<PSFGenerationService> psf_service_;
    std::unique_ptr<DeconvolutionService> deconv_service_;
    
    // Flag to track if service layer is initialized
    bool service_layer_initialized_;


    // Multithreading
    // std::unique_ptr<ThreadPool> background_pool_;
    // std::unordered_map<std::string, std::future<void>> running_operations_;
    // std::mutex operations_mutex_;
};