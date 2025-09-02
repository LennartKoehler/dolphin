#include "psf/configs/PSFConfig.h"
#include "psf/PSFGeneratorFactory.h"

PSFConfig::PSFConfig(const PSFConfig& other) 
    : sizeX(other.sizeX)
    , sizeY(other.sizeY) 
    , sizeZ(other.sizeZ)
    , resLateral_nm(other.resLateral_nm)
    , resAxial_nm(other.resAxial_nm)
    , NA(other.NA){
}

bool PSFConfig::loadFromJSON(const json& jsonData){
    try{// Load basic PSF dimensions (required)
        sizeX = readParameter<int>(jsonData, "sizeX");
        sizeY = readParameter<int>(jsonData, "sizeY");
        sizeZ = readParameter<int>(jsonData, "sizeZ");
        resAxial_nm = readParameter<int>(jsonData, "resAxial[nm]");
        resLateral_nm = readParameter<int>(jsonData, "resLateral[nm]");
        NA = readParameter<double>(jsonData, "NA");
        psfCubeVec = readParameter<std::vector<int>>(jsonData, "subimages");
        psfLayerVec = readParameter<std::vector<int>>(jsonData, "layers");

        return true;
    } catch (const json::exception &e) {
        std::cerr << "[ERROR] Invalid PSF JSON structure: " << e.what() << std::endl;
        return false;
    }
    loadFromJSONSpecific(jsonData);
    return true;
}
    

bool PSFConfig::compareDim(const PSFConfig &other) {
    if(this->sizeX != other.sizeX || this->sizeY != other.sizeY || this->sizeZ != other.sizeZ) {
        std::cerr << "[ERROR] All PSFs have to be the same size" << std::endl;
        return false;
    }
    return true;
}


std::shared_ptr<PSFConfig> PSFConfig::createFromJSON(const json& jsonData){
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<PSFConfig> config = factory.createConfig(jsonData);
    return config;

}

