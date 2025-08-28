#include "frontend/gui/uipsf/UIConfigPSFGibsonLanni.h"


UIConfigPSFGibsonLanni::UIConfigPSFGibsonLanni(){
    setParameters(std::make_shared<GibsonLanniPSFConfig>());
}

void UIConfigPSFGibsonLanni::setParameters(const std::shared_ptr<const GibsonLanniPSFConfig> config){

    auto configCopy = std::make_shared<GibsonLanniPSFConfig>(*config);
    psfConfig = configCopy;

    setDefaultParameters(configCopy);
    setSpecificParameters(configCopy);
}

std::shared_ptr<PSFConfig> UIConfigPSFGibsonLanni::getConfig(){
    return psfConfig;
}


void UIConfigPSFGibsonLanni::setSpecificParameters(std::shared_ptr<GibsonLanniPSFConfig> cfg) {
    std::vector<ParameterDescription> temp = {
        {"Oversampling", ParameterType::Double, &cfg->OVER_SAMPLING, 1.0, 10.0},
        {"Lambda (nm)", ParameterType::Double, &cfg->lambda_nm, 200.0, 700.0},
        {"Accuracy", ParameterType::Int, &cfg->accuracy, 1, 10},
        {"Objective ti0 (nm)", ParameterType::Double, &cfg->ti0_nm, 0.0, 20000.0},
        {"Objective ti (nm)", ParameterType::Double, &cfg->ti_nm, 0.0, 20000.0},
        {"Immersion ni0", ParameterType::Double, &cfg->ni0, 1.0, 2.0},
        {"Immersion ni", ParameterType::Double, &cfg->ni, 1.0, 2.0},
        {"Coverslip tg0 (nm)", ParameterType::Double, &cfg->tg0_nm, 0.0, 5000.0},
        {"Coverslip tg (nm)", ParameterType::Double, &cfg->tg_nm, 0.0, 5000.0},
        {"Coverslip ng0", ParameterType::Double, &cfg->ng0, 1.0, 2.0},
        {"Coverslip ng", ParameterType::Double, &cfg->ng, 1.0, 2.0},
        {"Sample ns", ParameterType::Double, &cfg->ns, 1.0, 2.0},
        {"Particle axial position (nm)", ParameterType::Double, &cfg->particleAxialPosition_nm, -5000.0, 5000.0}
    };
    parameters.insert(parameters.end(), temp.begin(), temp.end());

}