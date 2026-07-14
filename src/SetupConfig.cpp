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
    : Config()  // Base class must be default-constructed (parameters must not be copied)
{
    // Copy all values first, then register parameters (which point to our own members)
    psfConfigPath = other.psfConfigPath;
    backend = other.backend;
    nThreads = other.nThreads;
    nWorkerThreads = other.nWorkerThreads;
    nIOThreads = other.nIOThreads;
    nDevices = other.nDevices;
    maxMemHost_byte = other.maxMemHost_byte;
    maxMemDevice_byte = other.maxMemDevice_byte;
    outputPath = other.outputPath;
    numReaderThreads = other.numReaderThreads;
    outputCompression = other.outputCompression;
    outputCompressionLevel = other.outputCompressionLevel;

    registerAllParameters();
}

SetupConfigPSF& SetupConfigPSF::operator=(const SetupConfigPSF& other) {
    if (this != &other) {
        psfConfigPath = other.psfConfigPath;
        backend = other.backend;
        nThreads = other.nThreads;
        nWorkerThreads = other.nWorkerThreads;
        nIOThreads = other.nIOThreads;
        nDevices = other.nDevices;
        maxMemHost_byte = other.maxMemHost_byte;
        maxMemDevice_byte = other.maxMemDevice_byte;
        outputPath = other.outputPath;
        numReaderThreads = other.numReaderThreads;
        outputCompression = other.outputCompression;
        outputCompressionLevel = other.outputCompressionLevel;

        parameters.clear();
        registerAllParameters();
    }
    return *this;
}




void SetupConfigPSF::registerAllParameters(){

    parameters.push_back({ParameterType::FilePath, &outputPath, "Output Path", false, "output", "-o,--output", "Output Path", true, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &psfConfigPath, "PSF Config Path", false, "psf_config_path", "-i,--psf_config_path", "PSF config path", true, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});

    parameters.push_back({ParameterType::FilePath, &backend, "Backend", true, "backend", "--backend", "Backend type", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Int, &nThreads, "Number of Threads", true, "n_threads", "--n_threads", "Number of threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nWorkerThreads, "Number of Worker Threads", true, "n_worker_threads", "--n_worker_threads", "Number of worker threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nIOThreads, "Number of IO Threads", true, "n_io_threads", "--n_io_threads", "Number of IO threads", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Int, &nDevices, "Number of Devices", true, "n_devices", "--n_devices", "Number of devices", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::Float, &maxMemHost_byte, "Max Memory Host (byte)", true, "max_mem__host_byte", "--max_mem_host_byte", "Maximum memory usage on the host", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Float, &maxMemDevice_byte, "Max Memory Device (byte)", true, "max_mem__device_byte", "--max_mem_device_byte", "Maximum memory usage on the device", false, false, 0.0, 0.0, nullptr});
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
    : SetupConfigPSF(other)  // Copy base class values; base registers its own parameters
{
    // Copy derived values, then re-register parameters for the full SetupConfig
    imagePath = other.imagePath;
    psfFilePaths = other.psfFilePaths;
    labeledImage = other.labeledImage;
    labelPSFMap = other.labelPSFMap;
    multiplePsfConfigPaths = other.multiplePsfConfigPaths;
    savePsf = other.savePsf;

    registerAllParameters();  // clears and re-registers with pointers to our own members
}


// Copy assignment operator
SetupConfig& SetupConfig::operator=(const SetupConfig& other) {
    if (this != &other) {
        // Copy all values (base + derived), then re-register parameters
        SetupConfigPSF::operator=(other);  // copies base values and re-registers base params

        imagePath = other.imagePath;
        psfFilePaths = other.psfFilePaths;
        labeledImage = other.labeledImage;
        labelPSFMap = other.labelPSFMap;
        multiplePsfConfigPaths = other.multiplePsfConfigPaths;
        savePsf = other.savePsf;

        parameters.clear();
        registerAllParameters();  // re-register with pointers to our own members
    }
    return *this;
}



// a bit different that setupConfigPSF (other values set as required)
void SetupConfig::registerAllParameters(){

    // Clear parameters registered by the base class constructor (SetupConfigPSF::registerAllParameters)
    // to avoid duplicates, since this override replaces them with deconvolution-specific variants.
    parameters.clear();

    parameters.push_back({ParameterType::FilePath, &imagePath, "Image Path", false, "image_path", "-i,--image_path", "Input image path", true, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &outputPath, "Output Path", false, "output", "-o,--output", "Output Path", true, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});

    parameters.push_back({ParameterType::Int, &numReaderThreads, "Number of Reader Threads", true, "num_reader_threads", "--num_reader_threads", "Number of TIFF reader threads (0=auto)", false, true, 0.0, 100.0, nullptr});
    parameters.push_back({ParameterType::String, &outputCompression, "Output Compression", true, "output_compression", "--output_compression", "TIFF compression scheme (none, lzw, deflate)", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Int, &outputCompressionLevel, "Output Compression Level", true, "output_compression_level", "--output_compression_level", "Compression level (-1=default, 1-9 for deflate)", false, true, -1.0, 9.0, nullptr});
    parameters.push_back({ParameterType::Bool, &savePsf, "Save PSF", true, "save_psf", "--save_psf", "Save used PSF", false, false, 0.0, 0.0, nullptr});


    parameters.push_back({ParameterType::VectorString, &psfFilePaths, "PSF File Paths", true, "psf_file_paths", "--psf_file_paths", "PSF file paths", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::VectorString, &multiplePsfConfigPaths, "Multiple PSF Config Paths", true, "multiple_psf_config_paths", "--multiple_psf_config_paths", "PSF config paths", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::FilePath, &psfConfigPath, "PSF Config Path", true, "psf_config_path", "--psf_config_path", "PSF config path", false, false, 0.0, 0.0, nullptr});
    // parameters.push_back({ParameterType::FilePath, &psfDirPath, "psf_dir_path", true, "psf_dir_path", "--psf_dir_path", "PSF directory path", false, false, 0.0, 0.0, nullptr});


    parameters.push_back({ParameterType::FilePath, &labeledImage, "Labeled Image", true, "labeled_image", "--labeled_image", "Labeled image path", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::String, &labelPSFMap, "Label PSF Map", true, "label_psf_map", "--label_psf_map", "Label PSF map path", false, false, 0.0, 0.0, nullptr});


}
