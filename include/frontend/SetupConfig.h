#pragma once
#include <vector>
#include "psf/configs/PSFConfig.h"

class DeconvolutionConfig;

struct SetupConfig : public Config{
    SetupConfig(){}
    SetupConfig(const SetupConfig& other);
    SetupConfig& operator=(const SetupConfig& other);

    bool loadFromJSON(const json& jsonData) override;
    static SetupConfig createFromJSONFile(const std::string& path);


    // Arguments
    std::string imagePath;
    std::string psfConfigPath;
    std::string psfFilePath;
    std::string psfDirPath;
    std::string outputDir;
    bool time = false; //show time
    bool sep = false; //save layer separate (TIF dir)
    bool savePsf = false; //save PSF
    bool showExampleLayers = false; //show random example layer of image and PSF
    bool printInfo = false; //show metadata of image
    bool saveSubimages = false;
    std::string gpu = "";
    // std::vector<int> layers;
    // std::vector<int> subimages;

    std::shared_ptr<DeconvolutionConfig> deconvolutionConfig;



};


