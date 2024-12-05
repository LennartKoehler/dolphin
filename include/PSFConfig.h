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
    std::string psfModel = "gauss";
    std::vector<int> psfLayers; //sub-image layers for PSF
    std::vector<int> psfCubes; //sub-images for PSF

    void loadFromJSON(const std::string &directoryPath);
};


