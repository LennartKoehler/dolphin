#pragma once
#include <vector>
#include "DeconvolutionConfig.h"
#include "psf/configs/PSFConfig.h"

enum class Application{
    deconvolution,
    psfgeneration
};

struct SetupConfig : public Config{
    SetupConfig(){}
    bool loadFromJSON(const json& jsonData) override;
    static SetupConfig createFromJSONFile(const std::string& path);


    // Arguments
    Application app;
    std::string imagePath;
    std::string psfConfigPath;
    std::string psfFilePath;
    std::string psfDirPath;
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
    std::shared_ptr<PSFConfig> psfConfig;



};

