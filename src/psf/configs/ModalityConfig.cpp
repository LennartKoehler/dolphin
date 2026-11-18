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

#include "dolphin/psf/configs/ModalityConfig.h"

#include <spdlog/spdlog.h>

WidefieldModalityConfig::WidefieldModalityConfig(){
    registerAllParameters();
}

WidefieldModalityConfig::WidefieldModalityConfig(const WidefieldModalityConfig& other){
    // No members to copy yet; keep the registry in sync with the type.
    registerAllParameters();
}

void WidefieldModalityConfig::registerAllParameters(){
    // Widefield currently adds no parameters of its own; new widefield
    // parameters belong here and are forwarded into the owning PSFConfig.
}

ConfocalModalityConfig::ConfocalModalityConfig(){
    registerAllParameters();
}

ConfocalModalityConfig::ConfocalModalityConfig(const ConfocalModalityConfig& other){
    copyMembers(other);
}

void ConfocalModalityConfig::copyMembers(const ConfocalModalityConfig& other){
    emissionWavelength_nm = other.emissionWavelength_nm;
    excitationWavelength_nm = other.excitationWavelength_nm;
    pinholeRadius_nm = other.pinholeRadius_nm;
    registerAllParameters();
}

void ConfocalModalityConfig::registerAllParameters(){
    // Confocal-specific parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    getParameters().push_back({ParameterType::Float, &emissionWavelength_nm, "Emission Wavelength (nm)", true, "emission_wavelength_nm", "--emission_wavelength_nm", "Emission (detection) wavelength (confocal)", false, true, 200.0, 700.0, nullptr});
    getParameters().push_back({ParameterType::Float, &excitationWavelength_nm, "Excitation Wavelength (nm)", true, "excitation_wavelength_nm", "--excitation_wavelength_nm", "Excitation wavelength (confocal)", false, true, 200.0, 700.0, nullptr});
    getParameters().push_back({ParameterType::Float, &pinholeRadius_nm, "Pinhole Radius (nm)", true, "pinhole_radius_nm", "--pinhole_radius_nm", "Pinhole radius at the sample plane (reporting only)", false, true, 0.0, 10000.0, nullptr});
}

std::unique_ptr<ModalityConfig> makeModalityConfig(MicroscopeModality modality){
    switch (modality) {
        case MicroscopeModality::CONFOCAL:
            return std::make_unique<ConfocalModalityConfig>();
        case MicroscopeModality::WIDEFIELD:
            break;
    }
    return std::make_unique<WidefieldModalityConfig>();
}
