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

#include "dolphin/PSFGenerationService.h"
#include "dolphin/Logging.h"
#include "dolphin/PSFCreator.h"
#include "dolphin/ProgressTracking.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/ThreadPool.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <memory>
#include <spdlog/spdlog.h>

PSFGenerationService::PSFGenerationService()
    : initialized_(false){}

PSFGenerationService::~PSFGenerationService() {
    shutdown();
}

void PSFGenerationService::initialize() {
    if (initialized_) return;

    try {


        // Initialize generator factory
        generator_factory_ = &(PSFGeneratorFactory::getInstance());


        default_output_path_ = getExecutableDirectory() + "/results/";

        initialized_ = true;
        logger_->info("PSF Generation Service initialized successfully");

    } catch (const std::exception& e) {
        initialized_ = false;
        throw;
    }
}

bool PSFGenerationService::isInitialized() const {
    return initialized_;
}

void PSFGenerationService::shutdown() {
    if (!initialized_) return;

    generator_factory_ = nullptr;  // Raw pointer, just set to nullptr
    initialized_ = false;
    logger_->debug("PSF Generation Service shut down successfully");
}

std::unique_ptr<PSFGenerationResult> PSFGenerationService::generatePSF(const PSFGenerationRequest& request) {
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        if (!initialized_) {
            throw std::runtime_error("PSF Generation Service not initialized");
        }
        Logging::setFrontendLogCallback(request.getFrontendLogging());

        logger_->info("Generating PSF with request");

        // PSF Generation logic based on request type
        std::shared_ptr<PSF> psf;
        std::shared_ptr<SetupConfigPSF> setupConfig = request.getConfig();

        std::shared_ptr<ThreadPool> localThreadPool = std::make_shared<ThreadPool>(setupConfig->nThreads);

        // Check PSF config path
        if (!request.getConfig()->psfConfigPath.empty()) {
            logger_->info("Generating PSF from config file path: " + request.getConfig()->psfConfigPath);
            psf = createPSFFromFilePathInternal(request.getConfig()->psfConfigPath, request.getProgressCallback(), localThreadPool);
        }

        if (!psf) {
            return createResult(false, "Failed to create PSF",
                              std::chrono::duration<double>::zero());
        }
        std::string output_file;

        // Handle saving if requested
        if (!request.getConfig()->outputPath.empty()) {
            output_file = savePSF(request.getConfig()->outputPath, psf);

            // // if a config was not provided by file the generated psfconfig is saved next
            // // to the psf otherwise you already have the config somewhere, no need to save it
            // if (request.config_.psf_config_ != nullptr){
            //     std::string configFilename = "Config_" + filenameBase + ".json";
            //     std::string output_file_config = savePSFConfig(request.output_path, configFilename, request.config_.psf_config_);
            // }
        }



        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        auto result = createResult(true, "PSF generation completed successfully", duration);
        result->psf = psf;
        result->generated_path = output_file;

        logger_->flush();

        Logging::resetFrontendLogCallback();
        return result;

    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        std::string error_msg = "PSF generation failed: " + std::string(e.what());
        logger_->error(error_msg);
        return createResult(false, error_msg, duration);
    }
}

// std::future<std::unique_ptr<PSFGenerationResult>> PSFGenerationService::generatePSFAsync(const PSFGenerationRequest& request){
//     return thread_pool_->enqueue([this, request](){
//         return generatePSF(request);
//     });
// }

std::vector<std::string> PSFGenerationService::getSupportedPSFTypes() const {
    return supported_types_;
}

bool PSFGenerationService::validateConfig(const json& config) const {
    if (!config.is_object()) {
        return false;
    }

    // Basic validation - check for required fields if present
    if (config.contains("type")) {
        std::string type = config["type"];
        auto it = std::find(supported_types_.begin(), supported_types_.end(), type);
        if (it == supported_types_.end()) {
            return false;
        }
    }

    return true;
}

std::unique_ptr<PSFGenerationResult> PSFGenerationService::createResult(
    bool success,
    const std::string& message,
    std::chrono::duration<double> duration) {

    auto result = std::make_unique<PSFGenerationResult>(success, message, duration);
    return result;
}

std::unique_ptr<PSF> PSFGenerationService::createPSFFromConfigInternal(
    std::shared_ptr<PSFConfig> psfConfig,
    progressCallbackFn fn,
    std::shared_ptr<ThreadPool> threadPool) {
    try {
        logger_->info("Creating PSF from config using PSFConfig");
        return std::make_unique<PSF>(PSFCreator::generatePSFFromPSFConfig(psfConfig, threadPool, fn));
    } catch (const std::exception& e) {
        std::string error_msg = "Failed to create PSF from config: " + std::string(e.what());
        logger_->error(error_msg);
        throw;
    }
}

std::unique_ptr<PSF> PSFGenerationService::createPSFFromFilePathInternal(
    const std::string& path,
    progressCallbackFn fn,
    std::shared_ptr<ThreadPool> threadPool) {
    try {
        logger_->info("Creating PSF from file path using PSFCreator: " + path);
        std::shared_ptr<PSFConfig> config = PSFCreator::generatePSFConfigFromConfigPath(path);
        return createPSFFromConfigInternal(config, fn, threadPool);
    } catch (const std::exception& e) {

        std::string error_msg = "Failed to create PSF from file: " + std::string(e.what());
        logger_->error(error_msg);
        throw std::runtime_error(error_msg);
    }
}


bool PSFGenerationService::isValidPSFType(const std::string& psf_type) const {
    auto it = std::find(supported_types_.begin(), supported_types_.end(), psf_type);
    return it != supported_types_.end();
}

std::string PSFGenerationService::savePSF(const std::string& path, std::shared_ptr<PSF> psf){
    // Use filesystem::path for better path handling
    std::filesystem::path output_path = path;
    // std::filesystem::path output_path = base_path / filename;  // Automatically handles separators

    // Ensure directory exists
    std::filesystem::create_directories(output_path.parent_path());

    std::string output_path_str = output_path.string();

    psf->writeToTiffFile(output_path_str);
    logger_->info("PSF saved to: " + output_path_str);
    return output_path_str;
}

std::string PSFGenerationService::savePSFConfig(const std::string& path, const std::string& filename, std::shared_ptr<PSFConfig> psfconfig){
    // Use filesystem::path for better path handling
    std::filesystem::path base_path = path.empty() ? default_output_path_ : path;
    std::filesystem::path output_path = base_path / filename;  // Automatically handles separators

    // Ensure directory exists
    std::filesystem::create_directories(output_path.parent_path());

    std::string output_path_str = output_path.string();
    ordered_json jsonConfig = psfconfig->writeToJSON();
    std::ofstream o(output_path_str);
    o << std::setw(4) << jsonConfig << std::endl;
    logger_->info("PSFConfig saved to: " + output_path_str);
    return output_path_str;
}



std::string PSFGenerationService::getExecutableDirectory() {
    try {
        // Get the path of the current executable
        std::filesystem::path execPath = std::filesystem::canonical("/proc/self/exe");
        return execPath.parent_path().parent_path().string();
    } catch (const std::exception& e) {
        // Fallback to current working directory
        return std::filesystem::current_path().string();
    }
}







