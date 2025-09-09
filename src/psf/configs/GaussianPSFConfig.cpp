#include "psf/configs/GaussianPSFConfig.h"
#include <fstream>
#include <iostream>

#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

std::string GaussianPSFConfig::getName() const{
    return this->psfModelName;
}

GaussianPSFConfig::GaussianPSFConfig() : PSFConfig() {
    registerAllParameters();
}

GaussianPSFConfig::GaussianPSFConfig(const GaussianPSFConfig& other)
    : PSFConfig(other){
    registerAllParameters();
    qualityFactor = other.qualityFactor;
    sigmaX = other.sigmaX;
    sigmaY = other.sigmaY;
    sigmaZ = other.sigmaZ;
    nanometerScale = other.nanometerScale;
    pixelScaling = other.pixelScaling;
    psfLayers = other.psfLayers;
    psfCubes = other.psfCubes;
}


double GaussianPSFConfig::convertResolution(double resolution_nm){
    return convertSigma(resolution_nm * nanometerScale / pixelScaling);
}

double GaussianPSFConfig::convertSigma(double sigma){
    return sigma * qualityFactor;
}

void GaussianPSFConfig::registerAllParameters(){
    bool optional = true;
    
    // Set PSF model name
    psfModelName = "Gaussian";
    
    // Register Gaussian-specific parameters
    registerParameter("qualityFactor", qualityFactor, optional);
    registerParameter("sigmaX", sigmaX, optional);  // Optional because can be calculated from resolution
    registerParameter("sigmaY", sigmaY, optional);
    registerParameter("sigmaZ", sigmaZ, optional);
    registerParameter("layers", psfLayers, optional);
    registerParameter("subimages", psfCubes, optional);
    registerParameter("nanometerScale", nanometerScale, optional);
    registerParameter("pixelScaling", pixelScaling, optional);
}