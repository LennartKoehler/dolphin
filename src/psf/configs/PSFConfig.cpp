/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include <spdlog/spdlog.h>

PSFConfig::PSFConfig(){
    registerAllParameters();
}

PSFConfig::PSFConfig(const PSFConfig& other) 
    : Config()  // Delegate to default constructor first (registers parameters)
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
        spdlog::error("All PSFs have to be the same size");
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
    parameters.push_back({ParameterType::Float, &resLateral_nm, "resLateral_nm", false, "resLateral_nm", "--resLateral_nm", "Lateral resolution in nm", false, true, 10.0, 500.0, nullptr});
    parameters.push_back({ParameterType::Float, &resAxial_nm, "resAxial_nm", false, "resAxial_nm", "--resAxial_nm", "Axial resolution in nm", false, true, 50.0, 2000.0, nullptr});
    parameters.push_back({ParameterType::String, &ID, "ID", true, "ID", "--ID", "PSF identifier", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::String, &psfModelName, "modelName", true, "modelName", "--modelName", "PSF model name", false, false, 0.0, 0.0, nullptr});
}
