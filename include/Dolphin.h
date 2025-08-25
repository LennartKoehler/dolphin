#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#include "HyperstackImage.h"
#include "BaseDeconvolutionAlgorithm.h"
#include "psf/configs/PSFConfig.h"
#include "psf/generators/BasePSFGenerator.h"
#include "ConfigManager.h"


using json = nlohmann::json;


class Dolphin{
public:
    Dolphin();
    ~Dolphin();

    void init(int argc, char** argv);
    void run();


    

private:
    Hyperstack initHyperstack() const ;
    std::unique_ptr<BaseDeconvolutionAlgorithm> initDeconvolution(const std::vector<std::vector<int>>& psfCubeVec, const std::vector<std::vector<int>>& psfLayerVec);
    std::unique_ptr<BaseAlgorithm> setAlgorithm(const std::string& algorithmName);
    void initPSFs();

    std::unique_ptr<BasePSFGenerator> initPSFGenerator(const std::string& modelName, const json& PSFConfig);
    void addPSFConfigFromJSON(const json& configJson);
    void createPSFFromConfig(std::unique_ptr<PSFConfig> psfConfig);
    void createPSFFromFile(const std::string& path);



    ConfigManager config;
    std::vector<std::unique_ptr<PSFConfig>> psfConfigs;
    std::vector<std::unique_ptr<BasePSFGenerator>> psfGenerators;


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