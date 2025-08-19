#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../lib/CLI/CLI11.hpp"

#include "HyperstackImage.h"
#include "BaseDeconvolutionAlgorithm.h"
#include "psf/PSFConfig.h"
#include "psf/BasePSFGenerator.h"
#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"


using json = nlohmann::json;


class Dolphin{
public:
    Dolphin();
    ~Dolphin();

    bool init(int argc, char** argv);
    void run();
    void setCLIOptions();
    void setCuda();
    void initPSFs();
    Hyperstack initHyperstack() const ;
    std::unique_ptr<BaseDeconvolutionAlgorithm> initDeconvolution(const std::vector<std::vector<int>>& psfCubeVec, const std::vector<std::vector<int>>& psfLayerVec);
    std::unique_ptr<BaseAlgorithm> setAlgorithm(const std::string& algorithmName);
    std::unique_ptr<BasePSFGenerator> initPSFGenerator(const std::string& modelName, const json& PSFConfig);
    
    void createPSFFromConfig(std::unique_ptr<PSFConfig> psfConfig);
    void createPSFFromFile(const std::string& path);
    void handleInput();
    void handleJSONConfigs(const std::string& configPath);
    json loadJSONFile(const std::string& path) const;
    std::string extractImagePath(const json& file) const;
    void processPSFPaths();
    void processSinglePSFPath(const std::string& path);
    void processPSFPathArray();
    bool isJSONFile(const std::string& path);
    void addPSFConfigFromJSON(const json& config);
    void extractAlgorithmParameters();
    void extractOptionalParameters();
    void handleCLIConfigs();
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
    std::vector<std::unique_ptr<PSFConfig>> psfConfigs;
    std::vector<std::unique_ptr<BasePSFGenerator>> psfGenerators;
    std::vector<std::string> psfPaths;
    std::vector<std::string> psfPathsCLI;
    

    std::string gpu = "";

    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    CLI::Option_group* cli_group;

    std::vector<std::string> deconvolutionAlgorithmNames{
        "InverseFilter",
        "RichardsonLucy",
        "RichardsonLucyTotalVariation",
        "RegularizedInverseFilter",
    };
    std::vector<std::string> PSFGeneratorNames{
        "Gaussian",
        "GibsonLanni"
    };


    std::vector<std::vector<int>> psfCubeVec, psfLayerVec;
    std::vector<PSF> psfs;
    Hyperstack inputHyperstack;
    std::unique_ptr<BaseAlgorithm> algorithm;

};