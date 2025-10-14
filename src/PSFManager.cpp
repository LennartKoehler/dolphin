#include "PSFManager.h"
#include "psf/PSFGeneratorFactory.h"
#include <fstream>
#include <cassert>



std::shared_ptr<PSFConfig> PSFManager::generatePSFConfigFromConfigPath(const std::string& psfConfigPath){
    if (!isJSONFile(psfConfigPath)){
        throw std::runtime_error("PSF Config file is not a JSON file: " + psfConfigPath);
    }
    json config = loadJSONFile(psfConfigPath);
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<PSFConfig> psfConfig = factory.createConfig(config);
    return psfConfig;
}

PSF PSFManager::readPSFFromFilePath(const std::string& psfFilePath){
    PSF psf;
    psf.readFromTifFile(psfFilePath.c_str());
    return psf;
}

PSF PSFManager::generatePSFFromPSFConfig(std::shared_ptr<PSFConfig> psfConfig, ThreadPool* threadPool){

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<BasePSFGenerator> psfGenerator = factory.createGenerator(psfConfig);
    psfGenerator->setThreadPool(threadPool);
    PSF psf = psfGenerator->generatePSF();
    psf.ID = psfConfig->ID;
    return psf;
    
}

std::vector<std::shared_ptr<PSFConfig>> PSFManager::generatePSFsFromDir(const std::string& psfDirPath){
    std::vector<std::shared_ptr<PSFConfig>> psfs;
    
    try {
        // Check if directory exists
        if (!std::filesystem::exists(psfDirPath)) {
            throw std::runtime_error("Directory does not exist: " + psfDirPath);
        }
        
        if (!std::filesystem::is_directory(psfDirPath)) {
            throw std::runtime_error("Path is not a directory: " + psfDirPath);
        }
        
        std::cout << "[STATUS] Reading PSF configs from directory: " << psfDirPath << std::endl;
        
        // Iterate through all files in directory
        for (const auto& entry : std::filesystem::directory_iterator(psfDirPath)) {
            if (entry.is_regular_file()) {
                std::string filePath = entry.path().string();
                
                // Check if file is a JSON file
                if (isJSONFile(filePath)) {
                    try {
                        std::cout << "[STATUS] Processing config file: " << entry.path().filename().string() << std::endl;
                        
                        // Generate PSF from this config file
                        std::shared_ptr<PSFConfig> psf = generatePSFConfigFromConfigPath(filePath);
                        psfs.push_back(std::move(psf));
                        
                    } catch (const std::exception& e) {
                        std::cerr << "[WARNING] Failed to generate PSF from " << filePath 
                                  << ": " << e.what() << std::endl;
                        // Continue processing other files instead of stopping
                    }
                } else {
                    std::cout << "[INFO] Skipping non-JSON file: " << entry.path().filename().string() << std::endl;
                }
            }
        }
        
        if (psfs.empty()) {
            throw std::runtime_error("No valid PSF config files found in directory: " + psfDirPath);
        }
        
        std::cout << "[STATUS] Generated " << psfs.size() << " PSF(s) from directory" << std::endl;
        
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Filesystem error while reading directory " + psfDirPath + ": " + e.what());
    }
    
    return psfs;
}
bool PSFManager::isJSONFile(const std::string& path) {
    return path.substr(path.find_last_of(".") + 1) == "json";
}

json PSFManager::loadJSONFile(const std::string& filePath){
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + filePath);
    }
    

    json jsonFile;
    try{
        file >> jsonFile;
    }
    catch (const std::exception& e){
        throw std::runtime_error("PSFManager failed to read json");
    }
    return jsonFile;
}



// PSFPackage PSFManager::generatePackageFromPSFConfig(std::shared_ptr<PSFConfig> config){
//     PSFPackage package;
//     package.psfLayerVec.push_back(config->psfLayerVec);
//     package.psfCubeVec.push_back(config->psfCubeVec);
//     package.psfs.push_back(generatePSFFromPSFConfig(config));
//     return package;
// }



// PSFPackage PSFManager::generatePackage(const std::string& psfConfigPath){ 
//     PSFPackage psfpackage;    
//     if (isJSONFile(psfConfigPath)) {
//         json configJson = loadJSONFile(psfConfigPath);
//         if (configJson.contains("path") && configJson["path"].get<std::string>() != "") {
//             psfpackage.push_back(fromConfigPath(configJson["path"].get<std::string>()));
//         }
//         else{
//             psfpackage.push_back(fromConfig(configJson));
//         }
//     } else {
//         // if (!psfConfigPath.empty()) { // LK why do i need this? only add psfConfigPaths if theyre not empty?
//         psfpackage.push_back(fromFilePath(psfConfigPath));
//         // }
//     }
//     return psfpackage;
// }

// PSFPackage PSFManager::fromFilePath(const std::string& psfPath){
//     PSFPackage psfpackage;
//     PSF psftmp;
//     if (psfPath.substr(psfPath.find_last_of(".") + 1) == "tif" || psfPath.substr(psfPath.find_last_of(".") + 1) == "tiff" || psfPath.substr(psfPath.find_last_of(".") + 1) == "ometif") {
//         psftmp.readFromTifFile(psfPath.c_str());
//     } else {
//         psftmp.readFromTifDir(psfPath.c_str());
//     }
//     psfpackage.psfs.push_back(psftmp);
//     psfpackage.psfCubeVec.push_back({});
//     psfpackage.psfLayerVec.push_back({});
//     return psfpackage;
// }



// PSFPackage PSFManager::fromConfig(const json& configJson) {
//     PSFPackage psfpackage;

//     // // LK TODO i dont know what psfcubevec and layervec are and where they should be processed. the deconvolution algorithms rely on them. are they needed to create the psfs? do they need to be apart of PSFConfig?
//     psfpackage.psfCubeVec.push_back(configJson["subimages"].get<std::vector<int>>());
//     psfpackage.psfLayerVec.push_back(configJson["layers"].get<std::vector<int>>());

//     std::shared_ptr<PSFConfig> psfConfig = PSFConfig::createFromJSON(configJson);
    
//     psfpackage.psfs.push_back(createfromConfig(psfConfig));
//     return psfpackage;
// }









// //TODO
// PSFPackage PSFManager::fromDirPath(const std::string& psfDirPath) {
//     for (const auto& element : psfDirPath) {
//         // if (element.is_string()) {
//         //     std::string elementStr = element.get<std::string>();
//         //     processSinglePSFPath(elementStr);
//         // }
//     }
// }



// void PSFManager::PSFDimensionCheck(const PSFPackage& psfpackage){
//     int firstPsfX = psfpackage.psfs[0].image.slices[0].cols;
//     int firstPsfY = psfpackage.psfs[0].image.slices[0].rows;
//     int firstPsfZ = psfpackage.psfs[0].image.slices.size();
//     for (int i = 0; i < psfpackage.psfs.size(); i++) {
//         if(firstPsfX != psfpackage.psfs[i].image.slices[0].cols || firstPsfY != psfpackage.psfs[i].image.slices[0].rows || firstPsfZ != psfpackage.psfs[i].image.slices.size()) {
//             throw std::runtime_error("PSF sizes do not match");
//             std::cout << firstPsfX << " " << firstPsfY << " " << firstPsfZ << " " << psfpackage.psfs[i].image.slices[0].cols << " " << psfpackage.psfs[i].image.slices[0].rows << " "<<psfpackage.psfs[i].image.slices.size()<<std::endl;
            
//         }
//     }
//     std::cout << "[INFO] " << psfpackage.psfs.size() << " PSF(s) loaded" << std::endl;
// }



