#include "psf/configs/PSFConfig.h"
#include "psf/PSFGeneratorFactory.h"

PSFConfig::PSFConfig(){
    registerAllParameters();
}

PSFConfig::PSFConfig(const PSFConfig& other) 
    : Config(other)  // Delegate to default constructor first (registers parameters)
{
    // Then copy the values
    ID = other.ID;
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
    // Base PSF parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    parameters.push_back({ParameterType::Int, &sizeX, "sizeX", false, "sizeX", "--sizeX", "PSF size X", false, true, 1, 1024, nullptr});
    parameters.push_back({ParameterType::Int, &sizeY, "sizeY", false, "sizeY", "--sizeY", "PSF size Y", false, true, 1, 1024, nullptr});
    parameters.push_back({ParameterType::Int, &sizeZ, "sizeZ", false, "sizeZ", "--sizeZ", "PSF size Z", false, true, 1, 512, nullptr});
    parameters.push_back({ParameterType::Float, &NA, "NA", false, "NA", "--NA", "Numerical aperture", false, true, 0.1, 2.0, nullptr});
    parameters.push_back({ParameterType::Float, &resLateral_nm, "resLateral_nm", false, "resLateral[nm]", "--resLateral_nm", "Lateral resolution in nm", false, true, 10.0, 500.0, nullptr});
    parameters.push_back({ParameterType::Float, &resAxial_nm, "resAxial_nm", false, "resAxial[nm]", "--resAxial_nm", "Axial resolution in nm", false, true, 50.0, 2000.0, nullptr});
    parameters.push_back({ParameterType::String, &ID, "ID", true, "ID", "--ID", "PSF identifier", false, false, 0.0, 0.0, nullptr});
}
