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

#include <memory>

#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/psf/modality/ModalityStrategy.h"
#include "dolphin/psf/modality/PSFModel.h"

/**
 * Minimal PSF generator. The factory assembles it fully configured for ONE
 * PSF config: it holds the modality strategy (the orchestrator), the PSF
 * model, and the config itself. generatePSF() contains no logic of its own -
 * it runs the modality's generate with the held config, model and thread
 * pool (given via configure()).
 */
class PSFGenerator : public BasePSFGenerator {
public:
    PSFGenerator(std::shared_ptr<const PSFConfig> config,
                 const ModalityStrategy& modality,
                 std::shared_ptr<PSFModel> model);

    PSF generatePSF() const override;

    /** The generator is assembled for exactly one config (its modality and
     *  model were derived from it); a different config means a new generator
     *  from the factory. */
    void setConfig(const std::shared_ptr<const PSFConfig> config) override;
    bool hasConfig() override;

private:
    std::shared_ptr<const PSFConfig> config;
    const ModalityStrategy* modality;
    std::shared_ptr<PSFModel> model;
};
