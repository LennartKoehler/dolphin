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

#include <string>
#include "HelperClasses.h"
#include "Config.h"


enum class PaddingType{
    ZERO,
    MIRROR
};
class DeconvolutionConfig : public Config{
public:
    DeconvolutionConfig();
    DeconvolutionConfig(const DeconvolutionConfig& other);

    // Use the struct for parameters
    std::string algorithmName = "RichardsonLucyTotalVariation";
    int subimageSize = 0;
    int iterations = 10;
    float epsilon = 1e-6;
    float lambda = 0.001;
    PaddingType borderType = PaddingType::ZERO;
    std::string backenddeconv = "cpu";
    int nThreads = 1;
    float maxMem_GB = 0;
    bool verbose = false;
    RangeMap<std::string> cubePSFMap; // 
    RangeMap<std::string> layerPSFMap; // currently unused

    std::string deconvolutionType = "normal";




    virtual bool loadFromJSON(const json& jsonData) override;
    virtual json writeToJSON() override;

private:
    virtual void registerAllParameters();
};
