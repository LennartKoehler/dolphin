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

#pragma once
#include "dolphin/psf/configs/PSFConfig.h"




class GibsonLanniPSFConfig : public PSFConfig{
public:
	GibsonLanniPSFConfig();
	GibsonLanniPSFConfig(const GibsonLanniPSFConfig& other);


    float OVER_SAMPLING = 4.0f;
    float lambda_nm = 520.0f;
    int accuracy = 32;
    
    /** Working distance of the objective (design value). */
    float ti0_nm = 150000.0f;

    /** Working distance of the objective (experimental value). */
    float ti_nm = 150000.0f;

    /** Immersion medium refractive index (design value). */
    float ni0 = 1.515f;

    /** Immersion medium refractive index (experimental value). */
    float ni = 1.515f;

    /** Coverslip thickness (design value). */
    float tg0_nm = 170.0f;

    /** Coverslip thickness (experimental value). */
    float tg_nm = 170.0f;

    /** Coverslip refractive index (design value). */
    float ng0 = 1.5f;

    /** Coverslip refractive index (experimental value). */
    float ng = 1.5f;

    /** Sample refractive index. */
    float ns = 1.33f;

    /** Axial position of the particle. */
    float particleAxialPosition_nm = 1000.0f;

    /** Pixel size in axial direction. */
    float pixelSizeAxial_nm = 100.0f;

    /** Pixel size in lateral direction. */
    float pixelSizeLateral_nm = 100.0f;

    
private:
    void registerAllParameters();

};
