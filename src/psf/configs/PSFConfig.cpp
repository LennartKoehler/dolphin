#include "psf/configs/PSFConfig.h"
#include "psf/PSFGeneratorFactory.h"

PSFConfig::PSFConfig(){
    registerAllParameters();
}

PSFConfig::PSFConfig(const PSFConfig& other) 
    : Config(other)  // Delegate to default constructor first (registers parameters)
{
    // Then copy the values
    psfModelName = other.psfModelName;
    sizeX = other.sizeX;
    sizeY = other.sizeY;
    sizeZ = other.sizeZ;
    resLateral_nm = other.resLateral_nm;
    resAxial_nm = other.resAxial_nm;
    NA = other.NA;
    registerAllParameters();

    // Copy any other members
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

void PSFConfig::registerAllParameters(){
    bool optional = true;
    
    // Basic PSF dimensions (required)
    registerParameter("modelName", psfModelName, !optional);
    registerParameter("sizeX", sizeX, !optional);
    registerParameter("sizeY", sizeY, !optional);
    registerParameter("sizeZ", sizeZ, !optional);
    registerParameter("resAxial[nm]", resAxial_nm, !optional);
    registerParameter("resLateral[nm]", resLateral_nm, !optional);
    registerParameter("NA", NA, !optional);
    
    
    // Additional parameters that might be optional
    // Add any other PSF-specific parameters here
}
