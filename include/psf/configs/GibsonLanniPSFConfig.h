#pragma once
#include "PSFConfig.h"




class GibsonLanniPSFConfig : public PSFConfig{
public:
	GibsonLanniPSFConfig() {
	}


	GibsonLanniPSFConfig(const GibsonLanniPSFConfig& p)
	: PSFConfig(p){
		this->ng = p.ng;
		this->ng0 = p.ng0;
		this->ni = p.ni;
		this->ni0 = p.ni0;
		this->ns = p.ns;
		this->particleAxialPosition_nm = p.particleAxialPosition_nm;
		this->tg_nm = p.tg_nm;
		this->tg0_nm = p.tg0_nm;
		this->ti_nm = p.ti_nm;
		this->ti0_nm = p.ti0_nm;
		this->lambda_nm = p.lambda_nm;
        this->accuracy = p.accuracy;
        this->OVER_SAMPLING = p.OVER_SAMPLING;
	}
    bool loadFromJSONSpecific(const json& jsonData) override;
    void printValues() override;
	std::string getName() const override;



    std::string psfModelName = "GibsonLanni";
    
    double OVER_SAMPLING = 4.0;
    double lambda_nm = 520.0;
    int accuracy = 32;
    
    /** Working distance of the objective (design value). */
    double ti0_nm = 150000.0;

    /** Working distance of the objective (experimental value). */
    double ti_nm = 150000.0;

    /** Immersion medium refractive index (design value). */
    double ni0 = 1.515;

    /** Immersion medium refractive index (experimental value). */
    double ni = 1.515;

    /** Coverslip thickness (design value). */
    double tg0_nm = 170.0;

    /** Coverslip thickness (experimental value). */
    double tg_nm = 170.0;

    /** Coverslip refractive index (design value). */
    double ng0 = 1.5;

    /** Coverslip refractive index (experimental value). */
    double ng = 1.5;

    /** Sample refractive index. */
    double ns = 1.33;

    /** Axial position of the particle. */
    double particleAxialPosition_nm = 1000.0;


};
