#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#include "ServiceAbstractions.h"
#include "ServiceFactory.h"


using json = nlohmann::json;


class Dolphin{
public:
    Dolphin() = default;
    ~Dolphin(){}

    void init();

    std::unique_ptr<PSFGenerationResult> generatePSF(std::shared_ptr<PSFConfig> psfconfig);
    std::unique_ptr<PSFGenerationResult> generatePSF(const std::string& psfconfigpath);
    std::unique_ptr<DeconvolutionResult> deconvolve(std::shared_ptr<SetupConfig> setupConfig);
    std::shared_ptr<Hyperstack> convolve(const Hyperstack& image, std::shared_ptr<PSF> psf);


    

private:

    void setCuda();
    
    // Service layer components (abstracted)
    std::unique_ptr<ServiceFactory> service_factory_;
    std::unique_ptr<IPSFGenerationService> psf_service_;
    std::unique_ptr<IDeconvolutionService> deconv_service_;
    
    // Flag to track if service layer is initialized
    bool service_layer_initialized_;

};