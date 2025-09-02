#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#include "HyperstackImage.h"
#include "BaseDeconvolutionAlgorithm.h"
#include "psf/configs/PSFConfig.h"
#include "psf/generators/BasePSFGenerator.h"
#include "frontend/SetupConfig.h"
#include "DeconvolutionConfig.h"


using json = nlohmann::json;


class Dolphin{
public:
    Dolphin() = default;
    ~Dolphin(){}

    void init(SetupConfig* config);
    void run();


    

private:
    std::shared_ptr<PSF> generatePSF();
    std::shared_ptr<Hyperstack> deconvolve();
    std::shared_ptr<Hyperstack> convolve(const Hyperstack& image, std::shared_ptr<PSF> psf);

    void setCuda();
    Hyperstack initHyperstack() const ;
    std::shared_ptr<BaseDeconvolutionAlgorithm> initDeconvolutionAlgorithm(std::shared_ptr<DeconvolutionConfig> config, std::vector<std::vector<int>> psfCubeVec, std::vector<std::vector<int>> psfLayerVec);


    SetupConfig* config;

};