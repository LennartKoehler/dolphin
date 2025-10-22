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

#include "Dolphin.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>


#include <thread>
void Dolphin::init(){
    
    // Initialize service layer
    if (!service_layer_initialized_) {
        service_factory_ = ServiceFactory::create();
        
        psf_service_ = service_factory_->createPSFGenerationService();
        psf_service_->initialize();
        
        deconv_service_ = service_factory_->createDeconvolutionService();
        deconv_service_->initialize();
        
        service_layer_initialized_ = true;
        
        std::cout << "[INFO] Abstract service layer initialized successfully" << std::endl;
    }
}

std::unique_ptr<PSFGenerationResult> Dolphin::generatePSF(PSFGenerationRequest request){
    return psf_service_->generatePSF(request);
}

std::unique_ptr<DeconvolutionResult> Dolphin::deconvolve(DeconvolutionRequest request){
    return deconv_service_->deconvolve(request);
}

std::future<std::unique_ptr<PSFGenerationResult>> Dolphin::generatePSFAsync(PSFGenerationRequest request){
    return psf_service_->generatePSFAsync(request);
}

std::future<std::unique_ptr<DeconvolutionResult>> Dolphin::deconvolveAsync(DeconvolutionRequest request){
    return deconv_service_->deconvolveAsync(request);
}







// Hyperstack Dolphin::initHyperstack() const{
//     Hyperstack hyperstack;
//     if (config->imagePath.substr(config->imagePath.find_last_of(".") + 1) == "tif" || config->imagePath.substr(config->imagePath.find_last_of(".") + 1) == "tiff" || config->imagePath.substr(config->imagePath.find_last_of(".") + 1) == "ometif") {
//         hyperstack.readFromTifFile(config->imagePath.c_str());
//     } else {
//         std::cout << "[INFO] No file ending .tif, pretending image is DIR" << std::endl;
//         hyperstack.readFromTifDir(config->imagePath.c_str());
//     }

//     if (config->printInfo) {
//         hyperstack.printMetadata();
//     }
//     if (config->showExampleLayers) {
//         hyperstack.showChannel(0);
//     }
//     hyperstack.saveAsTifFile("../result/input_hyperstack.tif");
//     hyperstack.saveAsTifDir("../result/input_hyperstack");
//     return hyperstack;
// }




// void Dolphin::setCuda(){
//     if (config->gpu == "" || config->gpu == "none"){
//         return;
//     }
//     if (config->gpu != "cuda"){
//         throw std::runtime_error("Only cuda is supported, not: " + config->gpu);
//     }
// #ifdef CUDA_AVAILABLE
// #else
//     config->gpu = "";
// #endif
// }