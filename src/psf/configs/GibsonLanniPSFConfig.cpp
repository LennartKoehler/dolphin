#include "psf/configs/GibsonLanniPSFConfig.h"

GibsonLanniPSFConfig::GibsonLanniPSFConfig()
    : PSFConfig(){
    psfModelName = "GibsonLanni";
    registerAllParameters();
}

std::string GibsonLanniPSFConfig::getName() const {
    return this->psfModelName;
}


void GibsonLanniPSFConfig::registerAllParameters(){

    
    // GibsonLanni-specific parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    getParameters().push_back({ParameterType::Float, &ti0_nm, "ti0_nm", false, "ti0_nm", "--ti0_nm", "Working distance design", false, true, 0.0, 20000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ti_nm, "ti_nm", false, "ti_nm", "--ti_nm", "Working distance experimental", false, true, 0.0, 20000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ni0, "ni0", false, "ni0", "--ni0", "Immersion RI design", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ni, "ni", false, "ni", "--ni", "Immersion RI experimental", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &tg0_nm, "tg0_nm", false, "tg0_nm", "--tg0_nm", "Coverslip thickness design", false, true, 0.0, 5000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &tg_nm, "tg_nm", false, "tg_nm", "--tg_nm", "Coverslip thickness experimental", false, true, 0.0, 5000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ns, "ns", false, "ns", "--ns", "Sample RI", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &particleAxialPosition_nm, "particleAxialPosition_nm", false, "particleAxialPosition_nm", "--particleAxialPosition_nm", "Particle axial position", false, true, -5000.0, 5000.0, nullptr});
    getParameters().push_back({ParameterType::Float, &lambda_nm, "lambda_nm", false, "lambda_nm", "--lambda_nm", "Wavelength", false, true, 200.0, 700.0, nullptr});
    getParameters().push_back({ParameterType::Int, &accuracy, "accuracy", false, "accuracy", "--accuracy", "Accuracy", false, true, 1, 10, nullptr});
    getParameters().push_back({ParameterType::Float, &OVER_SAMPLING, "OVER_SAMPLING", false, "OVER_SAMPLING", "--OVER_SAMPLING", "Oversampling", false, true, 1.0, 10.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ng0, "ng0", true, "ng0", "--ng0", "Coverslip RI design", false, true, 1.0, 2.0, nullptr});
    getParameters().push_back({ParameterType::Float, &ng, "ng", true, "ng", "--ng", "Coverslip RI experimental", false, true, 1.0, 2.0, nullptr});
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
    ng0 = other.ng0;
    ng = other.ng;
    // dont clear because parent already cleared, else i clear the parent
    registerAllParameters();
}