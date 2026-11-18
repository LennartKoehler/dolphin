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

#include "dolphin/psf/generators/PSFGenerator.h"

#include <stdexcept>

PSFGenerator::PSFGenerator(std::shared_ptr<const PSFConfig> config,
                           const ModalityStrategy& modality,
                           std::shared_ptr<PSFModel> model)
    : config(std::move(config)), modality(&modality), model(std::move(model)) {}

bool PSFGenerator::hasConfig(){
    return config != nullptr;
}

void PSFGenerator::setConfig(const std::shared_ptr<const PSFConfig>){
    // The modality strategy and the model were derived from the config at
    // creation; silently swapping the config would desynchronize them.
    throw std::logic_error("PSFGenerator is bound to the config it was created with; create a new generator via the factory");
}

PSF PSFGenerator::generatePSF() const {
    if (!threadPool)
        throw std::runtime_error("PSFGenerator: no thread pool configured (call configure() first)");
    return modality->generate(*model, *config, *threadPool, progressTracker);
}
