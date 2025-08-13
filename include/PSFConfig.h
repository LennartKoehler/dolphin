#pragma once

#include <string>
#include <vector>

class PSFConfig {
public:
    int x = 20;
    int y = 20;
    int z = 40;
    double sigmax = 10;
    double sigmay = 10;
    double sigmaz = 10;
    double qualityFactor = 1.0; // 1.0 for perfect a perfect image, larger if the image is blurry and therefore the psd should be too
    double pixelScaling = 1e-6; //TODO should this be here? or in main config?
    std::string psfModel = "gauss";
    std::vector<int> psfLayers = {}; //sub-image layers for PSF
    std::vector<int> psfCubes = {}; //sub-images for PSF

    std::string psfPath = "";
    
    bool loadFromJSON(const std::string &directoryPath);
    bool compareDim(const PSFConfig &other);
    void printValues();
    double convertSigma(double sigma);
    double convertResolution(double resolution);

};


