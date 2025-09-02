#include "Dolphin.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ConvolutionAlgorithm.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "psf/PSFGeneratorFactory.h"
#include "PSFManager.h"
#include <sys/stat.h>
#ifdef CUDA_AVAILABLE
#include "../lib/cube/include/CUBE.h"
#endif

void Dolphin::init(SetupConfig* config){
    this->config = config;
}


void Dolphin::run(){
    setCuda();
    if (config->app == Application::deconvolution){
        deconvolve();
    }
    if (config->app == Application::psfgeneration){
        generatePSF();
    }
    

}



std::shared_ptr<PSF> Dolphin::generatePSF(){
    PSFManager psfmanager;
    PSF psf = psfmanager.generatePSF(config->psfConfigPath);
    return std::make_shared<PSF>(psf);
}

std::shared_ptr<Hyperstack> Dolphin::deconvolve(){
    PSFManager psfmanager; // TODO make singleton?
    PSFPackage psfpackage = psfmanager.handleSetupConfig(*config);
    std::shared_ptr<DeconvolutionConfig> deconvConfig = std::move(config->deconvolutionConfig);
    std::shared_ptr<BaseDeconvolutionAlgorithm> deconvAlgorithm = initDeconvolutionAlgorithm(std::move(deconvConfig), psfpackage.psfCubeVec, psfpackage.psfLayerVec);
    Hyperstack hyperstack = initHyperstack();
    Hyperstack result = deconvAlgorithm->run(hyperstack, psfpackage.psfs);

    // TODO write save function, maybe create more general writer class
    if (config->showExampleLayers) {
        result.showChannel(0);
    }

    result.saveAsTifFile("../result/deconv.tif");
    if(config->sep){
        result.saveAsTifDir("../result/deconv");
    }
    return std::make_unique<Hyperstack>(result);


}






// void Dolphin::savePSFs(){
//     for (int i = 0; i < psfs.size(); i++) {
//         psfs[i].saveAsTifFile("../result/psf_"+std::to_string(i)+".tif");
//     }
// }







Hyperstack Dolphin::initHyperstack() const{
    Hyperstack hyperstack;
    if (config->imagePath.substr(config->imagePath.find_last_of(".") + 1) == "tif" || config->imagePath.substr(config->imagePath.find_last_of(".") + 1) == "tiff" || config->imagePath.substr(config->imagePath.find_last_of(".") + 1) == "ometif") {
        hyperstack.readFromTifFile(config->imagePath.c_str());
    } else {
        std::cout << "[INFO] No file ending .tif, pretending image is DIR" << std::endl;
        hyperstack.readFromTifDir(config->imagePath.c_str());
    }

    if (config->printInfo) {
        hyperstack.printMetadata();
    }
    if (config->showExampleLayers) {
        hyperstack.showChannel(0);
    }
    hyperstack.saveAsTifFile("../result/input_hyperstack.tif");
    hyperstack.saveAsTifDir("../result/input_hyperstack");
    return hyperstack;
}



std::shared_ptr<BaseDeconvolutionAlgorithm> Dolphin::initDeconvolutionAlgorithm(std::shared_ptr<DeconvolutionConfig> config, const std::vector<std::vector<int>>& psfCubeVec, const std::vector<std::vector<int>>& psfLayerVec){


    DeconvolutionConfig deconvConfig;
    deconvConfig.psfCubeVec = psfCubeVec;
    deconvConfig.psfLayerVec = psfLayerVec;

    DeconvolutionAlgorithmFactory DAF = DeconvolutionAlgorithmFactory::getInstance();
    return DAF.create(config->algorithmName, deconvConfig);
}



void Dolphin::setCuda(){
    if (config->gpu == ""){
        return;
    }
    if (config->gpu != "cuda"){
        throw std::runtime_error("Only cuda is supported, not: " + config->gpu);
    }
#ifdef CUDA_AVAILABLE
#else
    config->gpu = "";
#endif
}