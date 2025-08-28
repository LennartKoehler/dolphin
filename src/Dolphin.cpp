#include "Dolphin.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ConvolutionAlgorithm.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "psf/PSFGeneratorFactory.h"

#include <sys/stat.h>
#ifdef CUDA_AVAILABLE
#include "../lib/cube/include/CUBE.h"
#endif





void Dolphin::init(ConfigManager* config){
    this->config = config;
    initPSFs();
    inputHyperstack = initHyperstack();
    algorithm = setAlgorithm(config->algorithmName);



}



void Dolphin::initPSFs(){
    for (const auto& psfJson : config->psfJSON){
        addPSFConfigFromJSON(psfJson);
    }

    for (const auto& psfPath : config->psfPaths) {
        createPSFFromFile(psfPath);
    }

    for (auto& psfConfig : psfConfigs) {
        createPSFFromConfig(std::move(psfConfig));
    }

    int firstPsfX = psfs[0].image.slices[0].cols;
    int firstPsfY = psfs[0].image.slices[0].rows;
    int firstPsfZ = psfs[0].image.slices.size();
    for (int i = 0; i < psfs.size(); i++) {
        if(firstPsfX != psfs[i].image.slices[0].cols || firstPsfY != psfs[i].image.slices[0].rows || firstPsfZ != psfs[i].image.slices.size()) {
            throw std::runtime_error("PSF sizes do not match");
            std::cout << firstPsfX << " " << firstPsfY << " " << firstPsfZ << " " << psfs[i].image.slices[0].cols << " " << psfs[i].image.slices[0].rows << " "<<psfs[i].image.slices.size()<<std::endl;
            
        }
    }
    std::cout << "[INFO] " << psfs.size() << " PSF(s) loaded" << std::endl;
    if (config->savePsf) {
        for (int i = 0; i < psfs.size(); i++) {
            psfs[i].saveAsTifFile("../result/psf_"+std::to_string(i)+".tif");
        }
    }
}

void Dolphin::createPSFFromConfig(std::unique_ptr<PSFConfig> psfConfig){

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::unique_ptr<BasePSFGenerator> psfGenerator = factory.createGenerator(std::move(psfConfig));
    PSF psftmp = psfGenerator->generatePSF();
    psfs.push_back(psftmp);
    
}

void Dolphin::addPSFConfigFromJSON(const json& configJson) {

    // LK TODO i dont know what psfcubevec and layervec are and where they should be processed. the deconvolution algorithms rely on them. are they needed to create the psfs? do they need to be apart of PSFConfig?
    psfCubeVec.push_back(configJson["subimages"].get<std::vector<int>>());
    psfLayerVec.push_back(configJson["layers"].get<std::vector<int>>());

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::unique_ptr<PSFConfig> psfConfig = factory.createConfig(configJson);
    psfConfigs.push_back(std::move(psfConfig));
    
}

void Dolphin::createPSFFromFile(const std::string& psfPath){
    PSF psftmp;
    if (psfPath.substr(psfPath.find_last_of(".") + 1) == "tif" || psfPath.substr(psfPath.find_last_of(".") + 1) == "tiff" || psfPath.substr(psfPath.find_last_of(".") + 1) == "ometif") {
        psftmp.readFromTifFile(psfPath.c_str());
    } else {
        psftmp.readFromTifDir(psfPath.c_str());
    }
    psfs.push_back(psftmp);
    psfCubeVec.push_back({});
    psfLayerVec.push_back({});
}

Hyperstack Dolphin::initHyperstack() const{
    Hyperstack hyperstack;
    if (config->image_path.substr(config->image_path.find_last_of(".") + 1) == "tif" || config->image_path.substr(config->image_path.find_last_of(".") + 1) == "tiff" || config->image_path.substr(config->image_path.find_last_of(".") + 1) == "ometif") {
        hyperstack.readFromTifFile(config->image_path.c_str());
    } else {
        std::cout << "[INFO] No file ending .tif, pretending image is DIR" << std::endl;
        hyperstack.readFromTifDir(config->image_path.c_str());
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

std::unique_ptr<BaseAlgorithm> Dolphin::setAlgorithm(const std::string& algorithmName){
    if (algorithmName == "Convolve"){ // TODO should be changed
        return std::make_unique<ConvolutionAlgorithm>();
    }
    else{
        return initDeconvolution(psfCubeVec, psfLayerVec);
    }
}

std::unique_ptr<BaseDeconvolutionAlgorithm> Dolphin::initDeconvolution(const std::vector<std::vector<int>>& psfCubeVec, const std::vector<std::vector<int>>& psfLayerVec){


    ConfigManager deconvConfig;
    deconvConfig.psfCubeVec = psfCubeVec;
    deconvConfig.psfLayerVec = psfLayerVec;

    deconvConfig.iterations = config->iterations;
    deconvConfig.epsilon = config->epsilon;
    deconvConfig.grid = config->grid;
    deconvConfig.lambda = config->lambda;
    deconvConfig.borderType = config->borderType;
    deconvConfig.psfSafetyBorder = config->psfSafetyBorder;
    deconvConfig.cubeSize = config->subimageSize;
    deconvConfig.gpu = config->gpu;

    deconvConfig.time = time;
    deconvConfig.saveSubimages = config->saveSubimages;

    DeconvolutionAlgorithmFactory DAF = DeconvolutionAlgorithmFactory::getInstance();
    return DAF.create(config->algorithmName, deconvConfig);
}

void Dolphin::run(){
    assert(algorithm && "deconvolutionAlgorithm must be initialized");
    assert(!psfs.empty() && "No PSFs loaded");
    assert(inputHyperstack.isValid() && "Input hyperstack not loaded");

    Hyperstack result = algorithm->run(inputHyperstack, psfs);
    // Hyperstack deconvHyperstack = deconvolutionAlgorithm->deconvolve(inputHyperstack, psfs);

    
    // TODO write save function, maybe create more general writer class
    if (config->showExampleLayers) {
        result.showChannel(0);
    }

    result.saveAsTifFile("../result/deconv.tif");
    if(config->sep){
        result.saveAsTifDir("../result/deconv");
    }

    std::cout << "[End DeconvTool]" << std::endl;
}