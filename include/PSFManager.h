#pragma once

#include <string>
#include "psf/configs/PSFConfig.h"
#include "psf/generators/BasePSFGenerator.h"
#include "frontend/SetupConfig.h"
#include "../lib/nlohmann/json.hpp"
using json = nlohmann::json;




// a bit weird because it does everything from loading to generating psf, but the input (through config or paths or directories makes it complicated)
// basically a namespace
// this thing is a mess
namespace PSFManager{

    PSF generatePSFFromConfigPath(const std::string& psfConfigPath);
    PSF generatePSFFromPSFConfig(std::shared_ptr<PSFConfig> config);
    std::vector<PSF> generatePSFsFromDir(const std::string& psfDirPath);


    json loadJSONFile(const std::string& path);
    bool isJSONFile(const std::string& path);
};

