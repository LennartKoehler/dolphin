#include "psf/configs/GaussianPSFConfig.h"
#include <fstream>
#include <iostream>

#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

std::string GaussianPSFConfig::getName() const{
    return this->psfModelName;
}

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


double GaussianPSFConfig::convertResolution(double resolution_nm){
    return convertSigma(resolution_nm * nanometerScale / pixelScaling);
}

double GaussianPSFConfig::convertSigma(double sigma){
    return sigma * qualityFactor;
}

void GaussianPSFConfig::registerAllParameters(){
    bool optional = true;
    
    // Register Gaussian-specific parameters
    registerParameter("qualityFactor", qualityFactor, optional);
    registerParameter("sigmaX", sigmaX, optional);  // Optional because can be calculated from resolution
    registerParameter("sigmaY", sigmaY, optional);
    registerParameter("sigmaZ", sigmaZ, optional);
    registerParameter("nanometerScale", nanometerScale, optional);
    registerParameter("pixelScaling", pixelScaling, optional);
}