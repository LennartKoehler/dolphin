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

#include "dolphin/Dolphin.h"
#include "dolphin/Logging.h"
#include <sys/stat.h>
#include <itkMultiThreaderBase.h>

void Dolphin::init(const std::filesystem::path& logDir){

    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);
    // Initialize service layer
    Logging::init(logDir);
    if (!service_layer_initialized_) {
        service_factory_ = ServiceFactory::create();

        psf_service_ = service_factory_->createPSFGenerationService();

        deconv_service_ = service_factory_->createDeconvolutionService();

        service_layer_initialized_ = true;
    }
}

std::unique_ptr<PSFGenerationResult> Dolphin::generatePSF(PSFGenerationRequest request){
    psf_service_->initialize();
    return psf_service_->generatePSF(request);
}

std::unique_ptr<DeconvolutionResult> Dolphin::deconvolve(DeconvolutionRequest request){
    deconv_service_->initialize();
    return deconv_service_->deconvolve(request);
}

