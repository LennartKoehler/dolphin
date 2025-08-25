#include "gui/uipsf/UIPSFConfigGibsonLanni.h"


UIPSFConfigGibsonLanni::UIPSFConfigGibsonLanni(){
    setParameters(std::make_shared<GibsonLanniPSFConfig>());
}

void UIPSFConfigGibsonLanni::setParameters(std::shared_ptr<PSFConfig> config){
    setConfig(config);
    setDefaultParameters(config);
    setGibsonLanniParameters(this->psfConfig);
}

std::shared_ptr<PSFConfig> UIPSFConfigGibsonLanni::getConfig(){
    return psfConfig;
}

void UIPSFConfigGibsonLanni::setConfig(std::shared_ptr<PSFConfig> config){
    auto g = std::dynamic_pointer_cast<GibsonLanniPSFConfig>(config);
    if (!g) {
        throw std::runtime_error("UIPSFConfigGibsonLanni: wrong config type passed!");
    }
    psfConfig = g;
}

void UIPSFConfigGibsonLanni::setGibsonLanniParameters(std::shared_ptr<GibsonLanniPSFConfig> cfg) {
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
    params.insert(params.end(), temp.begin(), temp.end());

}