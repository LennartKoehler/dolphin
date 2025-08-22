#pragma once


#include "PSF.h"
#include "PSFConfig.h"
#include "BasePSFGenerator.h"

class GaussianPSFConfig;

class GaussianPSFGenerator : public BasePSFGenerator {
public:
    GaussianPSFGenerator() = default;
    GaussianPSFGenerator(std::unique_ptr<PSFConfig>&& config) { setConfig(std::move(config)); }

    PSF generatePSF() const override;
    void setConfig(std::unique_ptr<PSFConfig> config) override;
    bool hasConfig() override;

private:
    std::unique_ptr<GaussianPSFConfig> config;
};





class GaussianPSFConfig : public PSFConfig{
public:
    std::string getName() override;
    bool loadFromJSON(const json& jsonData) override;
    void printValues() override;


    double convertSigma(double sigma);
    double convertResolution(double resolution);



    int sizeX = 20;
    int sizeY = 20;
    int sizeZ = 40;
    double sigmaX = 10;
    double sigmaY = 10;
    double sigmaZ = 10;
    double qualityFactor = 1.0; // 1.0 for perfect a perfect image, larger if the image is blurry and therefore the psd should be too
    double pixelScaling = 1e-6; //TODO should this be here? or in main config?
    double nanometerScale = 1e-9;
    std::vector<int> psfLayers = {}; //sub-image layers for PSF
    std::vector<int> psfCubes = {}; //sub-images for PSF

    std::string psfPath = "";
    std::string psfModelName = "Gaussian";
};