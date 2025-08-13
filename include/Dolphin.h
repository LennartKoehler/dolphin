#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../lib/CLI/CLI11.hpp"

#include "HyperstackImage.h"
#include "PSF.h"
#include "PSFConfig.h"
#include "PSFGenerator.h"
#include "GaussianPSFGeneratorAlgorithm.h"
#include "SimpleGaussianPSFGeneratorAlgorithm.h"
#include "BornWolfModel.h"
#include "BaseDeconvolutionAlgorithm.h"

#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"


using json = nlohmann::json;


struct PSFPackage{
    std::vector<std::vector<int>> psfCubeVec;
    std::vector<std::vector<int>> psfLayerVec;
    std::vector<PSF> psfs;
};

class Dolphin{
public:
    Dolphin();
    ~Dolphin();

    bool init(int argc, char** argv);
    void run();
    void setCLIOptions();
    void handleConfigs();
    void setCuda();
    PSFPackage initPSF();
    Hyperstack initHyperstack();
    std::unique_ptr<BaseAlgorithm> initDeconvolution(const std::vector<std::vector<int>>& psfCubeVec, const std::vector<std::vector<int>>& psfLayerVec);
    std::unique_ptr<BaseAlgorithm> algorithmFactory(const std::string& algorithmName);
    

private:
    // Arguments
    std::string image_path;
    std::string psf_path;
    std::string config_file_path;
    std::string algorithmName;
    int iterations = 10; //RL and RLTV
    double lambda = 0.01; //RIF and RLTV
    double epsilon = 1e-6; // complex divison
    bool time = false; //show time
    bool sep = false; //save layer separate (TIF dir)
    bool savePsf = false; //save PSF
    bool showExampleLayers = false; //show random example layer of image and PSF
    bool printInfo = false; //show metadata of image
    bool grid = false; //do grid processing
    int subimageSize = 0; //sub-image size (edge)
    int psfSafetyBorder = 10; //padding around PSF
    int borderType = cv::BORDER_REFLECT; //extension type of image
    bool saveSubimages = false;

    json config;
    PSFConfig psfconfig_1;
    PSFConfig psfconfig_2;
    std::vector<PSFConfig> psfConfigs;
    std::vector<std::string> psfPaths;
    std::vector<std::string> psfPathsCLI;
    

    std::string gpu = "";

    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    CLI::Option_group* cli_group;


    std::vector<std::vector<int>> psfCubeVec, psfLayerVec;
    std::vector<PSF> psfs;
    Hyperstack inputHyperstack;
    std::unique_ptr<BaseAlgorithm> algorithm;

};