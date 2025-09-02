#pragma once

#include <string>
#include "psf/configs/PSFConfig.h"
#include "psf/generators/BasePSFGenerator.h"
#include "frontend/SetupConfig.h"
#include "../lib/nlohmann/json.hpp"
using json = nlohmann::json;

struct PSFPackage{
    std::vector<PSF> psfs;
    std::vector<std::vector<int>> psfCubeVec, psfLayerVec;
    void push_back(const PSFPackage& other);
};


// a bit weird because it does everything from loading to generating psf, but the input (through config or paths or directories makes it complicated)
// basically a namespace
class PSFManager{
public:
    PSFManager() = default;
    PSFPackage handleSetupConfig(const SetupConfig& setupConfig);
    PSFPackage PSFFromConfigPath(const std::string& psfConfigPath);
    PSF generatePSF(const std::string& psfConfigPath);

private:
    void PSFDimensionCheck(const PSFPackage& psfs);
    PSF createPSFFromConfig(std::shared_ptr<PSFConfig> config);

    PSFPackage PSFFromPSFConfig(std::shared_ptr<PSFConfig> config);
    PSFPackage PSFFromFilePath(const std::string& psfFilePath);
    PSFPackage PSFFromDirPath(const std::string& psfDirPath);

    json loadJSONFile(const std::string& path) const;

    bool isJSONFile(const std::string& path);
    PSFPackage PSFFromConfig(const json& config);


};

