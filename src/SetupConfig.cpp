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

#include "dolphin/SetupConfig.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/backend/BackendFactory.h"

SetupConfigPSF::SetupConfigPSF() {
    registerAllParameters();
}


SetupConfigPSF SetupConfigPSF::createFromJSONFile(const std::string& filePath) {
    json jsonData = loadJSONFile(filePath);

    jsonData.erase("deconvolution_config"); // it can be part of one json file but should not be read as such
    SetupConfigPSF config;
    if (!config.loadFromJSON(jsonData)) {
        throw std::runtime_error("Failed to parse config file: " + filePath);
    }

    return config;
}

SetupConfigPSF::SetupConfigPSF(const SetupConfigPSF& other)
    : Config()  // Copy base class
{
    // First, register all parameters to set up the infrastructure
    registerAllParameters();

    // Then copy all the values
    psfConfigPath = other.psfConfigPath;
    // psfDirPath = other.psfDirPath;

    backend = other.backend;
    nThreads = other.nThreads;
    nWorkerThreads = other.nWorkerThreads;
    nIOThreads = other.nIOThreads;
    nDevices = other.nDevices;
    maxMem_GB = other.maxMem_GB;

    outputPath = other.outputPath;
}


// Copy assignment operator (recommended to implement if you have copy constructor)
SetupConfigPSF& SetupConfigPSF::operator=(const SetupConfigPSF& other) {
    if (this != &other) {  // Self-assignment check
        Config::operator=(other);  // Copy base class

        psfConfigPath = other.psfConfigPath;
        // psfDirPath = other.psfDirPath;

        backend = other.backend;
        nThreads = other.nThreads;
        nWorkerThreads = other.nWorkerThreads;
        nIOThreads = other.nIOThreads;
        nDevices = other.nDevices;
        maxMem_GB = other.maxMem_GB;

        outputPath = other.outputPath;
    }
    return *this;
}




void SetupConfigPSF::registerAllParameters(){

    parameters.push_back({ParameterType::FilePath, &outputPath, "Output Path", true, "output", "-o,--output", "Output Path", true, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &psfConfigPath, "PSF Config Path", true, "psf_config_path", "-i,--psf_config_path", "PSF config path", true, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});

    parameters.push_back({ParameterType::FilePath, &backend, "Backend", true, "backend", "--backend", "Backend type", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Int, &nThreads, "Number of Threads", false, "n_threads", "--n_threads", "Number of threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nWorkerThreads, "Number of Worker Threads", true, "n_worker_threads", "--n_worker_threads", "Number of worker threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nIOThreads, "Number of IO Threads", true, "n_io_threads", "--n_io_threads", "Number of IO threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nDevices, "Number of Devices", true, "n_devices", "--n_devices", "Number of devices", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Float, &maxMem_GB, "Max Memory (GB)", false, "max_mem_gb", "--max_mem_gb", "Maximum memory usage", false, false, 0.0, 0.0, nullptr});
}




SetupConfig::SetupConfig() {
    registerAllParameters();
}


SetupConfig SetupConfig::createFromJSONFile(const std::string& filePath) {
    json jsonData = loadJSONFile(filePath);

    jsonData.erase("deconvolution_config"); // it can be part of one json file but should not be read as such
    SetupConfig config;
    if (!config.loadFromJSON(jsonData)) {
        throw std::runtime_error("Failed to parse config file: " + filePath);
    }

    return config;
}

SetupConfig::SetupConfig(const SetupConfig& other)
    : SetupConfigPSF(other)  // Copy base class
{
    registerAllParameters();

    imagePath = other.imagePath;
    psfFilePath = other.psfFilePath;

    labeledImage = other.labeledImage;
    labelPSFMap = other.labelPSFMap;
    multiplePsfConfigPaths = other.multiplePsfConfigPaths;
    savePsf = other.savePsf;
}


// Copy assignment operator (recommended to implement if you have copy constructor)
SetupConfig& SetupConfig::operator=(const SetupConfig& other) {
    if (this != &other) {  // Self-assignment check
        SetupConfigPSF::operator=(other);  // Copy base class

        imagePath = other.imagePath;
        psfFilePath = other.psfFilePath;
        // psfDirPath = other.psfDirPath;

        labeledImage = other.labeledImage;
        labelPSFMap = other.labelPSFMap;
        multiplePsfConfigPaths = other.multiplePsfConfigPaths;

        savePsf = other.savePsf;
    }
    return *this;
}



// a bit different that setupConfigPSF (other values set as required)
void SetupConfig::registerAllParameters(){

    // Clear parameters registered by the base class constructor (SetupConfigPSF::registerAllParameters)
    // to avoid duplicates, since this override replaces them with deconvolution-specific variants.
    parameters.clear();

    parameters.push_back({ParameterType::FilePath, &outputPath, "Output Path", true, "output", "-o,--output", "Output Path", true, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::VectorString, &psfConfigPath, "PSF Config Path", true, "psf_config_path", "--psf_config_path", "PSF config path", false, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});

    parameters.push_back({ParameterType::FilePath, &backend, "Backend", true, "backend", "--backend", "Backend type", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Int, &nThreads, "Number of Threads", false, "n_threads", "--n_threads", "Number of threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nWorkerThreads, "Number of Worker Threads", true, "n_worker_threads", "--n_worker_threads", "Number of worker threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nIOThreads, "Number of IO Threads", true, "n_io_threads", "--n_io_threads", "Number of IO threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nDevices, "Number of Devices", true, "n_devices", "--n_devices", "Number of devices", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Float, &maxMem_GB, "Max Memory (GB)", false, "max_mem_gb", "--max_mem_gb", "Maximum memory usage", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Bool, &savePsf, "Save PSF", false, "save_psf", "--save_psf", "Save used PSF", false, false, 0.0, 0.0, nullptr});


    parameters.push_back({ParameterType::FilePath, &imagePath, "Image Path", false, "image_path", "-i,--image_path", "Input image path", true, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::VectorString, &psfFilePath, "PSF File Path", true, "psf_file_path", "--psf_file_path", "PSF file path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::VectorString, &multiplePsfConfigPaths, "Multiple PSF Config Path", true, "multiple_psf_config_path", "--psf_config_paths", "PSF config paths", true, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});


    parameters.push_back({ParameterType::FilePath, &labeledImage, "Labeled Image", true, "labeled_image", "--labeled_image", "Labeled image path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::String, &labelPSFMap, "Label PSF Map", true, "label_psf_map", "--label_psf_map", "Label PSF map path", false, false, 0.0, 0.0, nullptr});


}
