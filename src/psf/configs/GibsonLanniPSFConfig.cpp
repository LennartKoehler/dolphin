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

    
    // GibsonLanni-specific parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    getParameters().push_back({ParameterType::Float, &ti0_nm, "ti0_nm", false, "workingDistanceDesign[nm]", "--ti0_nm", "Working distance design", false, true, 0.0, 20000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ti_nm, "ti_nm", false, "workingDistanceExperimental[nm]", "--ti_nm", "Working distance experimental", false, true, 0.0, 20000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ni0, "ni0", false, "immersionRIDesign", "--ni0", "Immersion RI design", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ni, "ni", false, "immersionRIExperimental", "--ni", "Immersion RI experimental", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &tg0_nm, "tg0_nm", false, "coverslipThicknessDesign[nm]", "--tg0_nm", "Coverslip thickness design", false, true, 0.0, 5000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &tg_nm, "tg_nm", false, "coverslipThicknessExperimental[nm]", "--tg_nm", "Coverslip thickness experimental", false, true, 0.0, 5000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ns, "ns", false, "sampleRI", "--ns", "Sample RI", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &particleAxialPosition_nm, "particleAxialPosition_nm", false, "particleAxialPosition[nm]", "--particleAxialPosition_nm", "Particle axial position", false, true, -5000.0, 5000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &lambda_nm, "lambda_nm", false, "lambda[nm]", "--lambda_nm", "Wavelength", false, true, 200.0, 700.0, nullptr});
    getParameters().push_back({ParameterType::Int, &accuracy, "accuracy", false, "accuracy", "--accuracy", "Accuracy", false, true, 1, 10, nullptr});
    getParameters().push_back({ParameterType::Float, &OVER_SAMPLING, "OVER_SAMPLING", false, "OVER_SAMPLING", "--OVER_SAMPLING", "Oversampling", false, true, 1.0, 10.0, nullptr});
    getParameters().push_back({ParameterType::Float, &pixelSizeAxial_nm, "pixelSizeAxial_nm", false, "pixelSizeAxial[nm]", "--pixelSizeAxial_nm", "Pixel size axial", false, true, 1.0, 1000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &pixelSizeLateral_nm, "pixelSizeLateral_nm", false, "pixelSizeLateral[nm]", "--pixelSizeLateral_nm", "Pixel size lateral", false, true, 1.0, 1000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ng0, "ng0", true, "coverslipRIDesign", "--ng0", "Coverslip RI design", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ng, "ng", true, "coverslipRIExperimental", "--ng", "Coverslip RI experimental", false, true, 1.0, 2.0, nullptr});
}
GibsonLanniPSFConfig::GibsonLanniPSFConfig(const GibsonLanniPSFConfig& other)
    : PSFConfig(other){
    ti0_nm = other.ti0_nm;
    ti_nm = other.ti_nm;
    ni0 = other.ni0;
    ni = other.ni;
    tg0_nm = other.tg0_nm;
    tg_nm = other.tg_nm;
    ns = other.ns;
    particleAxialPosition_nm = other.particleAxialPosition_nm;
    lambda_nm = other.lambda_nm;
    accuracy = other.accuracy;
    OVER_SAMPLING = other.OVER_SAMPLING;
    pixelSizeAxial_nm = other.pixelSizeAxial_nm;
    pixelSizeLateral_nm = other.pixelSizeLateral_nm;
    ng0 = other.ng0;
    ng = other.ng;
    // dont clear because parent already cleared, else i clear the parent
    registerAllParameters();
}