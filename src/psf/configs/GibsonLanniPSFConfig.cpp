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

#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"

GibsonLanniPSFConfig::GibsonLanniPSFConfig()
    : PSFConfig(){
    psfModelName = "GibsonLanni";
    registerAllParameters();
}




void GibsonLanniPSFConfig::registerAllParameters(){
    PSFConfig::registerAllParameters();

    // GibsonLanni-specific parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    // Bounds accept everything physically meaningful: NA down to low-mag
    // objectives (0.01), refractive indices up to silicon (4.0), working
    // distances up to a few mm, wavelengths over the full glass/detector
    // transmission range. Anything tighter rejects real setups; nonsense
    // (<= 0, negative geometry) is rejected by GibsonLanniModel::validateConfig.
    getParameters().push_back({ParameterType::Float, &NA, "Numerical Aperture", false, "NA", "--NA", "Numerical aperture", false, true, 0.01, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ti0_nm, "Working Distance Design (nm)", false, "working_distance_design_nm", "--working_distance_design_nm", "Working distance design", false, true, 0.0, 4000000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ti_nm, "Working Distance Experimental (nm)", false, "working_distance_experimental_nm", "--working_distance_experimental_nm", "Working distance experimental", false, true, 0.0, 4000000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ni0, "Immersion RI Design", false, "immersion_ri_design", "--immersion_ri_design", "Immersion RI design", false, true, 1.0, 4.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ni, "Immersion RI Experimental", false, "immersion_ri_experimental", "--immersion_ri_experimental", "Immersion RI experimental", false, true, 1.0, 4.0, nullptr});
    getParameters().push_back({ParameterType::Float, &tg0_nm, "Coverslip Thickness Design (nm)", false, "coverslip_thickness_design_nm", "--coverslip_thickness_design_nm", "Coverslip thickness design", false, true, 0.0, 2000000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &tg_nm, "Coverslip Thickness Experimental (nm)", false, "coverslip_thickness_experimental_nm", "--coverslip_thickness_experimental_nm", "Coverslip thickness experimental", false, true, 0.0, 2000000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ns, "Sample RI", false, "sample_ri", "--sample_ri", "Sample RI", false, true, 1.0, 4.0, nullptr});
    getParameters().push_back({ParameterType::Float, &particleAxialPosition_nm, "Particle Axial Position (nm)", false, "particle_axial_position_nm", "--particle_axial_position_nm", "Particle axial position", false, true, -1000000.0, 1000000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &wavelength_nm, "Wavelength (nm)", false, "wavelength_nm", "--wavelength_nm", "Wavelength of the optical model (detection wavelength for widefield)", false, true, 100.0, 2000.0, nullptr});
    getParameters().push_back({ParameterType::Int, &integration_accuracy, "Integration Accuracy", false, "integration_accuracy", "--integration_accuracy", "Integration accuracy", false, true, 1, 10, nullptr});
    getParameters().push_back({ParameterType::Float, &pixelSizeAxial_nm, "Pixel Size Axial (nm)", false, "pixel_size_axial_nm", "--pixel_size_axial_nm", "Pixel size axial", false, true, 1.0, 100000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &pixelSizeLateral_nm, "Pixel Size Lateral (nm)", false, "pixel_size_lateral_nm", "--pixel_size_lateral_nm", "Pixel size lateral", false, true, 1.0, 100000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ng0, "Coverslip RI Design", true, "coverslip_ri_design", "--coverslip_ri_design", "Coverslip RI design", false, true, 1.0, 4.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ng, "Coverslip RI Experimental", true, "coverslip_ri_experimental", "--coverslip_ri_experimental", "Coverslip RI experimental", false, true, 1.0, 4.0, nullptr});
}
GibsonLanniPSFConfig::GibsonLanniPSFConfig(const GibsonLanniPSFConfig& other)
    : PSFConfig(other){
    NA = other.NA;
    ti0_nm = other.ti0_nm;
    ti_nm = other.ti_nm;
    ni0 = other.ni0;
    ni = other.ni;
    tg0_nm = other.tg0_nm;
    tg_nm = other.tg_nm;
    ns = other.ns;
    particleAxialPosition_nm = other.particleAxialPosition_nm;
    wavelength_nm = other.wavelength_nm;
    integration_accuracy = other.integration_accuracy;
    pixelSizeAxial_nm = other.pixelSizeAxial_nm;
    pixelSizeLateral_nm = other.pixelSizeLateral_nm;
    ng0 = other.ng0;
    ng = other.ng;
    // dont clear because parent already cleared, else i clear the parent
    registerAllParameters();
}
