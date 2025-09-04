
#pragma once
#include "PSFConfig.h"

class GaussianPSFConfig : public PSFConfig{
public:
    std::string getName() const override;
    bool loadFromJSONSpecific(const json& jsonData) override;
    void printValues() override;


    double convertSigma(double sigma);
    double convertResolution(double resolution);



    double sigmaX = 10;
    double sigmaY = 10;
    double sigmaZ = 10;
    double qualityFactor = 1.0; // 1.0 for perfect a perfect image, larger if the image is blurry and therefore the psd should be too
    double pixelScaling = 1e-6; //TODO should this be here? or in main config?
    double nanometerScale = 1e-9;
    std::vector<int> psfLayers = {}; //sub-image layers for PSF
    std::vector<int> psfCubes = {}; //sub-images for PSF

    std::string psfModelName = "Gaussian";
};