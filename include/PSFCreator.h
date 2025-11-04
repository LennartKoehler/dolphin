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
#include "psf/configs/PSFConfig.h"
#include "psf/generators/BasePSFGenerator.h"
#include "frontend/SetupConfig.h"
#include "../lib/nlohmann/json.hpp"
using json = nlohmann::json;



class ThreadPool;
// a bit weird because it does everything from loading to generating psf, but the input (through config or paths or directories makes it complicated)
// basically a namespace
// this thing is a mess
namespace PSFCreator{
    PSF readPSFFromFilePath(const std::string& psfFilePath);
    std::shared_ptr<PSFConfig> generatePSFConfigFromConfigPath(const std::string& psfConfigPath);
    PSF generatePSFFromPSFConfig(std::shared_ptr<PSFConfig> config, ThreadPool* threadpool);
    std::vector<std::shared_ptr<PSFConfig>> generatePSFsFromDir(const std::string& psfDirPath);


    json loadJSONFile(const std::string& path);
    bool isJSONFile(const std::string& path);
};

