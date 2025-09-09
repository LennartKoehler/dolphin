#include "psf/configs/GibsonLanniPSFConfig.h"

GibsonLanniPSFConfig::GibsonLanniPSFConfig() : PSFConfig(){
    registerAllParameters();
}

std::string GibsonLanniPSFConfig::getName() const {
    return this->psfModelName;
}


void GibsonLanniPSFConfig::registerAllParameters(){
    bool optional = true;
    psfModelName = "GibsonLanni";
    registerParameter("workingDistanceDesign[nm]", ti0_nm, !optional);
    registerParameter("workingDistanceExperimental[nm]", ti_nm, !optional);
    registerParameter("immersionRIDesign", ni0, !optional);
    registerParameter("immersionRIExperimental", ni, !optional);
    registerParameter("coverslipThicknessDesign[nm]", tg0_nm, !optional);
    registerParameter("coverslipThicknessExperimental[nm]", tg_nm, !optional);
    registerParameter("sampleRI", ns, !optional);
    registerParameter("particleAxialPosition[nm]", particleAxialPosition_nm, !optional);
    registerParameter("lambda[nm]", lambda_nm, !optional);
    registerParameter("accuracy", accuracy, !optional);
    registerParameter("OVER_SAMPLING", OVER_SAMPLING, !optional);
    registerParameter("coverslipRIDesign", ng0, optional);
    registerParameter("coverslipRIExperimental", ng, optional);

}
GibsonLanniPSFConfig::GibsonLanniPSFConfig(const GibsonLanniPSFConfig& other)
    : PSFConfig(other){
    registerAllParameters();
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
}