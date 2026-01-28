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
#include <vector>
#include "dolphin/Config.h"

class DeconvolutionConfig;

class SetupConfig : public Config{
public:
    SetupConfig();
    SetupConfig(const SetupConfig& other);

    std::string getName() const override { return std::string("SetupConfig"); };
    SetupConfig& operator=(const SetupConfig& other);

    bool loadFromJSON(const json& jsonData) override;
    static SetupConfig createFromJSONFile(const std::string& path);

    // Arguments
    std::string imagePath;
    std::string psfConfigPath;
    std::string psfFilePath;
    std::string psfDirPath;
    std::string outputDir;
    std::string labeledImage;
    std::string labelPSFMap;
    std::string strategyType = "normal";
    std::string backend = "libcpu_backend.so";

    int nThreads = 1;
    int nIOThreads = 1;
    int nWorkerThreads = 1;
    int nDevices = 1;
    float maxMem_GB = 1;


    // bool sep = false; //save layer separate (TIF dir)
    // bool savePsf = false; //save PSF
    // bool showExampleLayers = false; //show random example layer of image and PSF
    // bool printInfo = false; //show metadata of image
    // bool saveSubimages = false;

    std::shared_ptr<DeconvolutionConfig> deconvolutionConfig;

private:

    virtual void registerAllParameters();


};


