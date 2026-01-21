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
#include "PSFConfig.h"

class GaussianPSFConfig : public PSFConfig{
public:
    GaussianPSFConfig();
    GaussianPSFConfig(const GaussianPSFConfig& other);
    std::string getName() const override;


    float convertSigma(float sigma);
    float convertResolution(float resolution);


    float sigmaX = 10;
    float sigmaY = 10;
    float sigmaZ = 10;
    float qualityFactor = 1.0; // 1.0 for perfect a perfect image, larger if the image is blurry and therefore the psd should be too
    float pixelScaling = 1e-6; //TODO should this be here? or in main config?
    float nanometerScale = 1e-9;
    std::vector<int> psfLayers = {}; //sub-image layers for PSF
    std::vector<int> psfCubes = {}; //sub-images for PSF



private:
    void registerAllParameters();

};