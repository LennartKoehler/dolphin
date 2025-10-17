#pragma once
#include "PSFConfig.h"




class GibsonLanniPSFConfig : public PSFConfig{
public:
	GibsonLanniPSFConfig();
	GibsonLanniPSFConfig(const GibsonLanniPSFConfig& other);


	std::string getName() const override;


    float OVER_SAMPLING = 4.0;
    float lambda_nm = 520.0;
    int accuracy = 32;
    
    /** Working distance of the objective (design value). */
    float ti0_nm = 150000.0;

    /** Working distance of the objective (experimental value). */
    float ti_nm = 150000.0;

    /** Immersion medium refractive index (design value). */
    float ni0 = 1.515;

    /** Immersion medium refractive index (experimental value). */
    float ni = 1.515;

    /** Coverslip thickness (design value). */
    float tg0_nm = 170.0;

    /** Coverslip thickness (experimental value). */
    float tg_nm = 170.0;

    /** Coverslip refractive index (design value). */
    float ng0 = 1.5;

    /** Coverslip refractive index (experimental value). */
    float ng = 1.5;

    /** Sample refractive index. */
    float ns = 1.33;

    /** Axial position of the particle. */
    float particleAxialPosition_nm = 1000.0;

    
private:
    void registerAllParameters();

};
