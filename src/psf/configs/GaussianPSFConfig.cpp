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

#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

GaussianPSFConfig::GaussianPSFConfig() : PSFConfig() {
    psfModelName = "Gaussian";
    registerAllParameters();
}


GaussianPSFConfig::GaussianPSFConfig(const GaussianPSFConfig& other)
    : PSFConfig(other){
    qualityFactor = other.qualityFactor;
    sigmaX = other.sigmaX;
    sigmaY = other.sigmaY;
    sigmaZ = other.sigmaZ;
    nanometerScale = other.nanometerScale;
    pixelScaling = other.pixelScaling;
    psfLayers = other.psfLayers;
    psfCubes = other.psfCubes;
    // dont clear because parent already cleared
    registerAllParameters();

}


float GaussianPSFConfig::convertResolution(float resolution_nm){
    return convertSigma(resolution_nm * nanometerScale / pixelScaling);
}

float GaussianPSFConfig::convertSigma(float sigma){
    return sigma * qualityFactor;
}

void GaussianPSFConfig::registerAllParameters(){
     
    // Gaussian-specific parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    getParameters().push_back({ParameterType::Float, &qualityFactor, "qualityFactor", true, "quality_factor", "--qualityFactor", "Quality factor", false, true, 0.1, 10.0, nullptr});
    getParameters().push_back({ParameterType::Float, &sigmaX, "sigmaX", true, "sigma_x", "--sigmaX", "Sigma X", false, true, 0.1, 100.0, nullptr});
    getParameters().push_back({ParameterType::Float, &sigmaY, "sigmaY", true, "sigma_y", "--sigmaY", "Sigma Y", false, true, 0.1, 100.0, nullptr});
    getParameters().push_back({ParameterType::Float, &sigmaZ, "sigmaZ", true, "sigma_z", "--sigmaZ", "Sigma Z", false, true, 0.1, 100.0, nullptr});
    getParameters().push_back({ParameterType::Float, &nanometerScale, "nanometerScale", true, "nanometer_scale", "--nanometerScale", "Nanometer scale", false, true, 1e-9, 1e-6, nullptr});
    getParameters().push_back({ParameterType::Float, &pixelScaling, "pixelScaling", true, "pixel_scaling", "--pixelScaling", "Pixel scaling", false, true, 1e-9, 1e-4, nullptr});
    
    // Handle vector parameters separately since they're not directly supported by ConfigParameter
    // These will need special handling in loadFromJSON and writeToJSON methods
}
