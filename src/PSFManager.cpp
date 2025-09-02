#include "PSFManager.h"
#include "psf/PSFGeneratorFactory.h"
#include <fstream>



PSFPackage PSFManager::handleSetupConfig(const SetupConfig& setupConfig) {
    PSFPackage psfPackage;
    if(setupConfig.psfConfig != nullptr){
        psfPackage.push_back(PSFFromPSFConfig(setupConfig.psfConfig)); // TODO
    }

    if (!setupConfig.psfConfigPath.empty()) {
        psfPackage.push_back(PSFFromConfigPath(setupConfig.psfConfigPath));
    }
    
    if (!setupConfig.psfDirPath.empty()) {
        psfPackage.push_back(PSFFromDirPath(setupConfig.psfDirPath));
    } 
    
    if (!setupConfig.psfFilePath.empty()){
        psfPackage.push_back(PSFFromFilePath(setupConfig.psfFilePath));
    }
    PSFDimensionCheck(psfPackage);
    return psfPackage;
}



PSF PSFManager::generatePSF(const std::string& psfConfigPath){
    if (!isJSONFile(psfConfigPath)){
        throw std::runtime_error("PSF Config file is not a JSON file: " + psfConfigPath);
    }
    json config = loadJSONFile(psfConfigPath);
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<PSFConfig> psfConfig = factory.createConfig(config);
    
    PSF psf = createPSFFromConfig(std::move(psfConfig));
    return psf;
}


PSFPackage PSFManager::PSFFromPSFConfig(std::shared_ptr<PSFConfig> config){
    PSFPackage package;
    package.psfLayerVec.push_back(config->psfLayerVec);
    package.psfCubeVec.push_back(config->psfCubeVec);
    package.psfs.push_back(createPSFFromConfig(config));
    return package;
}



PSFPackage PSFManager::PSFFromConfigPath(const std::string& psfConfigPath){ 
    PSFPackage psfpackage;    
    if (isJSONFile(psfConfigPath)) {
        json configJson = loadJSONFile(psfConfigPath);
        if (configJson.contains("path") && configJson["path"].get<std::string>() != "") {
            psfpackage.push_back(PSFFromConfigPath(configJson["path"].get<std::string>()));
        }
        else{
            psfpackage.push_back(PSFFromConfig(configJson));
        }
    } else {
        // if (!psfConfigPath.empty()) { // LK why do i need this? only add psfConfigPaths if theyre not empty?
        psfpackage.push_back(PSFFromFilePath(psfConfigPath));
        // }
    }
}

PSFPackage PSFManager::PSFFromFilePath(const std::string& psfPath){
    PSFPackage psfpackage;
    PSF psftmp;
    if (psfPath.substr(psfPath.find_last_of(".") + 1) == "tif" || psfPath.substr(psfPath.find_last_of(".") + 1) == "tiff" || psfPath.substr(psfPath.find_last_of(".") + 1) == "ometif") {
        psftmp.readFromTifFile(psfPath.c_str());
    } else {
        psftmp.readFromTifDir(psfPath.c_str());
    }
    psfpackage.psfs.push_back(psftmp);
    psfpackage.psfCubeVec.push_back({});
    psfpackage.psfLayerVec.push_back({});
    return psfpackage;
}



PSFPackage PSFManager::PSFFromConfig(const json& configJson) {
    PSFPackage psfpackage;

    // // LK TODO i dont know what psfcubevec and layervec are and where they should be processed. the deconvolution algorithms rely on them. are they needed to create the psfs? do they need to be apart of PSFConfig?
    psfpackage.psfCubeVec.push_back(configJson["subimages"].get<std::vector<int>>());
    psfpackage.psfLayerVec.push_back(configJson["layers"].get<std::vector<int>>());

    std::shared_ptr<PSFConfig> psfConfig = PSFConfig::createFromJSON(configJson);
    
    psfpackage.psfs.push_back(createPSFFromConfig(std::move(psfConfig)));
    return psfpackage;
}

PSF PSFManager::createPSFFromConfig(std::shared_ptr<PSFConfig> psfConfig){

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<BasePSFGenerator> psfGenerator = factory.createGenerator(std::move(psfConfig));
    PSF psftmp = psfGenerator->generatePSF();
    return psftmp;
    
}


json PSFManager::loadJSONFile(const std::string& filePath) const {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + filePath);
    }
    
    std::cout << "[STATUS] " << filePath << " successfully read" << std::endl;

    json jsonFile;
    file >> jsonFile;
    return jsonFile;
}






//TODO
PSFPackage PSFManager::PSFFromDirPath(const std::string& psfDirPath) {
    for (const auto& element : psfDirPath) {
        // if (element.is_string()) {
        //     std::string elementStr = element.get<std::string>();
        //     processSinglePSFPath(elementStr);
        // }
    }
}

bool PSFManager::isJSONFile(const std::string& path) {
    return path.substr(path.find_last_of(".") + 1) == "json";
}


void PSFManager::PSFDimensionCheck(const PSFPackage& psfpackage){
    int firstPsfX = psfpackage.psfs[0].image.slices[0].cols;
    int firstPsfY = psfpackage.psfs[0].image.slices[0].rows;
    int firstPsfZ = psfpackage.psfs[0].image.slices.size();
    for (int i = 0; i < psfpackage.psfs.size(); i++) {
        if(firstPsfX != psfpackage.psfs[i].image.slices[0].cols || firstPsfY != psfpackage.psfs[i].image.slices[0].rows || firstPsfZ != psfpackage.psfs[i].image.slices.size()) {
            throw std::runtime_error("PSF sizes do not match");
            std::cout << firstPsfX << " " << firstPsfY << " " << firstPsfZ << " " << psfpackage.psfs[i].image.slices[0].cols << " " << psfpackage.psfs[i].image.slices[0].rows << " "<<psfpackage.psfs[i].image.slices.size()<<std::endl;
            
        }
    }
    std::cout << "[INFO] " << psfpackage.psfs.size() << " PSF(s) loaded" << std::endl;
}




// void Dolphin::createPSFFromConfig(std::shared_ptr<PSFConfig> psfConfig){

//     PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
//     std::shared_ptr<BasePSFGenerator> psfGenerator = factory.createGenerator(std::move(psfConfig));
//     PSF psftmp = psfGenerator->generatePSF();
//     psfs.push_back(psftmp);
    
// }

// void Dolphin::addPSFConfigFromJSON(const json& configJson) {

//     // LK TODO i dont know what psfcubevec and layervec are and where they should be processed. the deconvolution algorithms rely on them. are they needed to create the psfs? do they need to be apart of PSFConfig?
//     psfCubeVec.push_back(configJson["subimages"].get<std::vector<int>>());
//     psfLayerVec.push_back(configJson["layers"].get<std::vector<int>>());

//     PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
//     std::shared_ptr<PSFConfig> psfConfig = factory.createConfig(configJson);
//     psfConfigs.push_back(std::move(psfConfig));
    
// }

// void Dolphin::createPSFFromFile(const std::string& psfPath){
//     PSF psftmp;
//     if (psfPath.substr(psfPath.find_last_of(".") + 1) == "tif" || psfPath.substr(psfPath.find_last_of(".") + 1) == "tiff" || psfPath.substr(psfPath.find_last_of(".") + 1) == "ometif") {
//         psftmp.readFromTifFile(psfPath.c_str());
//     } else {
//         psftmp.readFromTifDir(psfPath.c_str());
//     }
//     psfs.push_back(psftmp);
//     psfCubeVec.push_back({});
//     psfLayerVec.push_back({});
// }


void PSFPackage::push_back(const PSFPackage& other){
    this->psfCubeVec.insert(psfCubeVec.end(), other.psfCubeVec.begin(), other.psfCubeVec.end());
    this->psfLayerVec.insert(psfLayerVec.end(), other.psfLayerVec.begin(), other.psfLayerVec.end());
    this->psfs.insert(psfs.end(), other.psfs.begin(), other.psfs.end());

}