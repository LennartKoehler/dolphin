#include "psf/GibsonLanniPSFGenerator.h"

std::string GibsonLanniPSFConfig::getName(){
    return this->psfModelName;
}

bool GibsonLanniPSFConfig::loadFromJSON(const json& jsonData) {
    try {
        psfModelName = "GibsonLanni";

        loadFromJSONBase(jsonData);
        
        // Load Gibson-Lanni specific parameters (required)
        ti0 = readParameter<double>(jsonData, "workingDistanceDesign");
        ti = readParameter<double>(jsonData, "workingDistanceExperimental");
        ni0 = readParameter<double>(jsonData, "immersionRIDesign");
        ni = readParameter<double>(jsonData, "immersionRIExperimental");
        tg0 = readParameter<double>(jsonData, "coverslipThicknessDesign");
        tg = readParameter<double>(jsonData, "coverslipThicknessExperimental");
        ns = readParameter<double>(jsonData, "sampleRI");
        particleAxialPosition = readParameter<double>(jsonData, "particleAxialPosition");
        lambda = readParameter<double>(jsonData, "lambda");
        accuracy = readParameter<int>(jsonData, "accuracy");
        OVER_SAMPLING = readParameter<double>(jsonData, "OVER_SAMPLING");

        // Optional parameters with defaults
        readParameterOptional<double>(jsonData, "coverslipRIDesign", ng0);
        readParameterOptional<double>(jsonData, "coverslipRIExperimental", ng);

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
    std::cout << "  Working distance (design): " << ti0 << " μm" << std::endl;
    std::cout << "  Working distance (experimental): " << ti << " μm" << std::endl;
    std::cout << "  Immersion RI (design): " << ni0 << std::endl;
    std::cout << "  Immersion RI (experimental): " << ni << std::endl;
    std::cout << "  Coverslip thickness (design): " << tg0 << " μm" << std::endl;
    std::cout << "  Coverslip thickness (experimental): " << tg << " μm" << std::endl;
    std::cout << "  Coverslip RI (design): " << ng0 << std::endl;
    std::cout << "  Coverslip RI (experimental): " << ng << std::endl;
    std::cout << "  Sample RI: " << ns << std::endl;
    std::cout << "  Particle axial position: " << particleAxialPosition << " μm" << std::endl;
    
    // Computational parameters
    std::cout << "  Wavelength: " << lambda << " nm" << std::endl;
    std::cout << "  Numerical Aperture: " << NA << std::endl;
    std::cout << "  Lateral resolution: " << resLateral_nm << " nm" << std::endl;
    std::cout << "  Axial resolution: " << resAxial_nm << " nm" << std::endl;
    std::cout << "  Accuracy: " << accuracy << std::endl;
    std::cout << "  Oversampling factor: " << OVER_SAMPLING << std::endl;
    
    // if (z > 0) {
    //     std::cout << "  Current Z-slice: " << z << std::endl;
    // }

    // // PSF layers and cubes (if applicable)
    // if (!psfLayers.empty()) {
    //     std::cout << "[STATUS] PSF layers: ";
    //     for (const int& layer : psfLayers) {
    //         std::cout << layer << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // if (!psfCubes.empty()) {
    //     std::cout << "[STATUS] PSF cubes: ";
    //     for (const int& cube : psfCubes) {
    //         std::cout << cube << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // PSF path (if loading from file)
    // if (!psfPath.empty()) {
    //     std::cout << "[INFO] Loading PSF from file: " << psfPath << std::endl;
    // }
}