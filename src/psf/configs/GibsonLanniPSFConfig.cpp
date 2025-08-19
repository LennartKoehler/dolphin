#include "psf/GibsonLanniPSFGenerator.h"

std::string GibsonLanniPSFConfig::getName(){
    return this->psfModelName;
}

bool GibsonLanniPSFConfig::loadFromJSON(const json& jsonData) {
    try {
        psfModelName = "GibsonLanni";

        loadFromJSONBase(jsonData);
        
        // Load Gibson-Lanni specific parameters (required)
        workingDistanceDesign = readParameter<double>(jsonData, "workingDistanceDesign");
        workingDistanceExperimental = readParameter<double>(jsonData, "workingDistanceExperimental");
        immersionRIDesign = readParameter<double>(jsonData, "immersionRIDesign");
        immersionRIExperimental = readParameter<double>(jsonData, "immersionRIExperimental");
        coverslipThicknessDesign = readParameter<double>(jsonData, "coverslipThicknessDesign");
        coverslipThicknessExperimental = readParameter<double>(jsonData, "coverslipThicknessExperimental");
        sampleRI = readParameter<double>(jsonData, "sampleRI");
        particleAxialPosition = readParameter<double>(jsonData, "particleAxialPosition");
        lambda = readParameter<double>(jsonData, "lambda");
        accuracy = readParameter<int>(jsonData, "accuracy");
        OVER_SAMPLING = readParameter<double>(jsonData, "OVER_SAMPLING");

        // Optional parameters with defaults
        readParameterOptional<double>(jsonData, "coverslipRIDesign", coverslipRIDesign);
        readParameterOptional<double>(jsonData, "coverslipRIExperimental", coverslipRIExperimental);

        return true;

    } catch (const json::exception& e) {
        std::cerr << "[ERROR] JSON parsing error in GibsonLanniPSFConfig: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Error loading GibsonLanniPSFConfig: " << e.what() << std::endl;
        return false;
    }
}

void GibsonLanniPSFConfig::printValues() {
    std::cout << "[INFO] Gibson-Lanni PSF parameters loaded from JSON file:" << std::endl;
    std::cout << "  Model: " << psfModelName << std::endl;
    std::cout << "  PSF Size: " << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
    
    // Optical parameters
    std::cout << "  Working distance (design): " << workingDistanceDesign << " μm" << std::endl;
    std::cout << "  Working distance (experimental): " << workingDistanceExperimental << " μm" << std::endl;
    std::cout << "  Immersion RI (design): " << immersionRIDesign << std::endl;
    std::cout << "  Immersion RI (experimental): " << immersionRIExperimental << std::endl;
    std::cout << "  Coverslip thickness (design): " << coverslipThicknessDesign << " μm" << std::endl;
    std::cout << "  Coverslip thickness (experimental): " << coverslipThicknessExperimental << " μm" << std::endl;
    std::cout << "  Coverslip RI (design): " << coverslipRIDesign << std::endl;
    std::cout << "  Coverslip RI (experimental): " << coverslipRIExperimental << std::endl;
    std::cout << "  Sample RI: " << sampleRI << std::endl;
    std::cout << "  Particle axial position: " << particleAxialPosition << " μm" << std::endl;
    
    // Computational parameters
    std::cout << "  Wavelength: " << lambda << " nm" << std::endl;
    std::cout << "  Numerical Aperture: " << NA << std::endl;
    std::cout << "  Lateral resolution: " << resLateral_nm << " nm" << std::endl;
    std::cout << "  Axial resolution: " << resAxial_nm << " nm" << std::endl;
    std::cout << "  Accuracy: " << accuracy << std::endl;
    std::cout << "  Oversampling factor: " << OVER_SAMPLING << std::endl;
}