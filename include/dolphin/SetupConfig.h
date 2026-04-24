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
#include <array>
#include "dolphin/deconvolution/DeconvolutionConfig.h"

class SetupConfigPSF : public Config{
public:
    SetupConfigPSF();
    SetupConfigPSF(const SetupConfigPSF& other);

    std::string getName() const override { return std::string("SetupConfigPSF"); };
    SetupConfigPSF& operator=(const SetupConfigPSF& other);

    static SetupConfigPSF createFromJSONFile(const std::string& path);

    std::string psfConfigPath;
    std::string outputPath;
    std::string backend = "cpu";

    int nThreads = 1;
    int nIOThreads = 1;
    int nWorkerThreads = 1;
    int nDevices = 1;
    float maxMem_GB = 1;

protected:

    virtual void registerAllParameters();
};



// for deconvolution
class SetupConfig : public SetupConfigPSF{
public:
    SetupConfig();
    SetupConfig(const SetupConfig& other);

    std::string getName() const override { return std::string("SetupConfig"); };
    SetupConfig& operator=(const SetupConfig& other);

    static SetupConfig createFromJSONFile(const std::string& path);

    std::string labeledImage;
    std::string labelPSFMap;
    std::string imagePath;
    std::vector<std::string> psfFilePath;
    std::vector<std::string> multiplePsfConfigPaths;
    bool savePsf = false;
private:
    virtual void registerAllParameters() override;
};

