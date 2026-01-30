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

#include "dolphin/ServiceFactory.h"
#include "dolphin/PSFGenerationService.h"
#include "dolphin/DeconvolutionService.h"
#include <spdlog/spdlog.h>

// Thread-local singleton instance using Meyer's pattern
thread_local static ServiceFactoryImpl* tls_instance = nullptr;

std::unique_ptr<PSFGenerationService> ServiceFactoryImpl::createPSFGenerationService() {
    auto service = std::make_unique<PSFGenerationService>();
    

    service->setLogger(spdlog::get("psf"));
    
    return service;
}

std::unique_ptr<DeconvolutionService> ServiceFactoryImpl::createDeconvolutionService() {
    auto service = std::make_unique<DeconvolutionService>();
    
    service->setLogger(spdlog::get("deconvolution"));

    
    return service;
}



ServiceFactoryImpl& ServiceFactoryImpl::getInstance() {
    if (!tls_instance) {
        tls_instance = new ServiceFactoryImpl();
    }
    return *tls_instance;
}

// Implementation of static method declared in ServiceAbstractions.h
ServiceFactory* ServiceFactory::create() {
    return &ServiceFactoryImpl::getInstance();
}