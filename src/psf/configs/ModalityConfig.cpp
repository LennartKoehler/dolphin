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
    pinholeRadius_um = other.pinholeRadius_um;
    registerAllParameters();
}

void ConfocalModalityConfig::registerAllParameters(){
    // Confocal-specific parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    // Wavelengths: [100, 2000] nm spans everything glass optics and
    // silicon/GaAs detectors can still transmit/detect - only typos and
    // unit errors land outside.
    getParameters().push_back({ParameterType::Float, &emissionWavelength_nm, "Emission Wavelength (nm)", true, "emission_wavelength_nm", "--emission_wavelength_nm", "Emission (detection) wavelength (confocal)", false, true, 100.0, 2000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &excitationWavelength_nm, "Excitation Wavelength (nm)", true, "excitation_wavelength_nm", "--excitation_wavelength_nm", "Excitation wavelength (confocal, <= 0 falls back to the emission wavelength)", false, true, -1000.0, 2000.0, nullptr});
    // 0 is the defined point-pinhole limit; large radii only degenerate the
    // confocal product toward widefield (plus convolution cost), so the cap
    // is generous.
    getParameters().push_back({ParameterType::Float, &pinholeRadius_um, "Pinhole Radius (um)", true, "pinhole_radius_um", "--pinhole_radius_um", "Pinhole radius at the sample plane in um (0 = point pinhole)", false, true, 0.0, 100.0, nullptr});
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
