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
#include "dolphin/ProgressTracking.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/SetupConfig.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;



class ThreadPool;
// a bit weird because it does everything from loading to generating psf, but the input (through config or paths or directories makes it complicated)
// basically a namespace
// this thing is a mess
namespace PSFCreator{
    std::vector<PSF> readPSFsFromFilePath(const std::vector<std::string>& psfFilePath);
    std::shared_ptr<PSFConfig> generatePSFConfigFromConfigPath(const std::string& psfConfigPath);
    PSF generatePSFFromPSFConfig(std::shared_ptr<PSFConfig> config, std::shared_ptr<ThreadPool> threadpool, progressCallbackFn fn);
    std::vector<std::shared_ptr<PSFConfig>> generatePSFsFromDir(const std::string& psfDirPath);


    // overrides the shape in the config. This is for when this psf is used in deconvolution it will be adjusted to the image shape
    std::vector<std::shared_ptr<PSFConfig>> generatePSFConfigsFromConfigPathWithShape(const std::vector<std::string>& paths, const CuboidShape& overrdeShape);
    std::vector<std::shared_ptr<PSFConfig>> generatePSFConfigsFromConfigPath(const std::vector<std::string>& paths);
    std::vector<std::string> stringSplit(const std::string& paths);

    json loadJSONFile(const std::string& path);
    bool isJSONFile(const std::string& path);
};

