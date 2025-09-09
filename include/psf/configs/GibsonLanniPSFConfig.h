#pragma once
#include "PSFConfig.h"




class GibsonLanniPSFConfig : public PSFConfig{
public:
	GibsonLanniPSFConfig();
	GibsonLanniPSFConfig(const GibsonLanniPSFConfig& other);


	std::string getName() const override;


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


private:
    virtual void registerAllParameters() override;
};
