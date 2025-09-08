#include "PSFGenerationService.h"
#include "PSFManager.h"
#include "psf/PSFGeneratorFactory.h"
#include "psf/configs/GaussianPSFConfig.h"
#include "psf/configs/GibsonLanniPSFConfig.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <memory>

PSFGenerationService::PSFGenerationService()
    : initialized_(false),
      logger_([](const std::string& msg) { std::cout << "[PSF_SERVICE] " << msg << std::endl; }),
      error_handler_([](const std::string& msg) { std::cerr << "[PSF_ERROR] " << msg << std::endl; })
{
    // Initialize supported PSF types
    supported_types_ = {"Gaussian", "GibsonLanni", "BornWolf"};
    
    // Initialize default logger and error handler
    default_logger_ = [](const std::string& msg) { std::cout << "[DEFAULT] " << msg << std::endl; };
    default_error_handler_ = [](const std::string& msg) { std::cerr << "[ERROR] " << msg << std::endl; };
}

PSFGenerationService::~PSFGenerationService() {
    shutdown();
}

void PSFGenerationService::initialize() {
    if (initialized_) return;
    
    try {

        
        // Initialize generator factory
        generator_factory_ = &(PSFGeneratorFactory::getInstance());
        
        // Create default config loader if not set
        if (!config_loader_) {
            config_loader_ = [](const std::string& path) {
                std::ifstream file(path);
                if (!file.is_open()) {
                    throw std::runtime_error("Failed to open config file: " + path);
                }
                json j;
                file >> j;
                return j;
            };
        }

        default_output_path_ = getExecutableDirectory() + "/results/";
        
        initialized_ = true;
        logMessage("PSF Generation Service initialized successfully");

    } catch (const std::exception& e) {
        handleError("Failed to initialize PSF Generation Service: " + std::string(e.what()));
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
    logMessage("PSF Generation Service shut down successfully");
}

std::unique_ptr<PSFGenerationResult> PSFGenerationService::generatePSF(const PSFGenerationRequest& request) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        if (!initialized_) {
            throw std::runtime_error("PSF Generation Service not initialized");
        }
        
        logMessage("Generating PSF with request");
        
        // PSF Generation logic based on request type
        std::shared_ptr<PSF> psf;
        
        // Check PSF config path
        if (!request.config_.config_path_.empty()) {
            logMessage("Generating PSF from config file path: " + request.config_.config_path_);
            psf = createPSFFromFilePathInternal(request.config_.config_path_);
        }
        // Check PSF config object
        else if (request.config_.psf_config_ != nullptr) {
            logMessage("Generating PSF from config object: " + request.config_.psf_config_->getName());
            psf = createPSFFromConfigInternal(request.config_.psf_config_);
        }
        else {
            throw std::runtime_error("No PSF configuration provided");
        }
        
        if (!psf) {
            return createResult(false, "Failed to create PSF",
                              std::chrono::duration<double>::zero());
        }
        std::string output_file;

        // Handle saving if requested
        if (request.save_result) {
            std::string filename = "PSF.tiff";
            output_file = saveResult(request.output_path, filename, psf);
        }
        
        if (request.show_example) {
            logMessage("Showing example PSF layer visualization");
            // psf->showChannel(0); // Uncomment when Hyperstack display is available
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        auto result = createResult(true, "PSF generation completed successfully", duration);
        result->psf = psf;
        result->generated_path = output_file;
        
        return result;
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::string error_msg = "PSF generation failed: " + std::string(e.what());
        logMessage(error_msg);
        return createResult(false, error_msg, duration);
    }
}

std::unique_ptr<PSFGenerationResult> PSFGenerationService::generatePSFFromConfig(std::shared_ptr<PSFConfig> config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {

        
        PSFGenerationRequest request;
        
        // Set up PSF config info with file path
        PSFGenerationRequest::PSFConfigInfo config_info;
        config_info.psf_config_ = config;
        request.setConfig(config_info);
        
        auto result = generatePSF(request);
        
        return result;
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::string error_msg = "PSF generation from config failed: " + std::string(e.what());
        logMessage(error_msg);
        return createResult(false, error_msg, duration);
    }
}

std::unique_ptr<PSFGenerationResult> PSFGenerationService::generatePSFFromFilePath(const std::string& path) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        PSFGenerationRequest request;
        
        // Set up PSF config info
        PSFGenerationRequest::PSFConfigInfo config_info;
        config_info.config_path_ = path;

        request.setConfig(config_info);
        
        return generatePSF(request);
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::string error_msg = "PSF generation from file failed: " + std::string(e.what());
        logMessage(error_msg);
        return createResult(false, error_msg, duration);
    }
}

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

void PSFGenerationService::setLogger(std::function<void(const std::string&)> logger) {
    logger_ = logger ? logger : default_logger_;
}

void PSFGenerationService::setConfigLoader(std::function<json(const std::string&)> loader) {
    config_loader_ = loader;
}

void PSFGenerationService::setDefaultLogger(std::function<void(const std::string&)> logger) {
    default_logger_ = logger;
    if (!logger_) {
        logger_ = default_logger_;
    }
}

void PSFGenerationService::setErrorHandler(std::function<void(const std::string&)> handler) {
    error_handler_ = handler ? handler : default_error_handler_;
}



void PSFGenerationService::logMessage(const std::string& message) {
    if (logger_) {
        logger_("INFO: " + message);
    }
}

void PSFGenerationService::handleError(const std::string& error) {
    if (error_handler_) {
        error_handler_("ERROR: " + error);
    }
}

std::unique_ptr<PSFGenerationResult> PSFGenerationService::createResult(
    bool success,
    const std::string& message,
    std::chrono::duration<double> duration) {
    
    auto result = std::make_unique<PSFGenerationResult>(success, message, duration);
    return result;
}

std::unique_ptr<PSF> PSFGenerationService::createPSFFromConfigInternal(std::shared_ptr<PSFConfig> psfConfig) {
    try {
        logMessage("Creating PSF from config using PSFManager");
        return std::make_unique<PSF>(PSFManager::generatePSFFromPSFConfig(psfConfig));
    } catch (const std::exception& e) {
        std::string error_msg = "Failed to create PSF from config: " + std::string(e.what());
        logMessage(error_msg);
        handleError(error_msg);
        throw std::runtime_error(error_msg);
    }
}

std::unique_ptr<PSF> PSFGenerationService::createPSFFromFilePathInternal(const std::string& path) {
    try {
        logMessage("Creating PSF from file path using PSFManager: " + path);
        return std::make_unique<PSF>(PSFManager::generatePSFFromConfigPath(path));
    } catch (const std::exception& e) {
        std::string error_msg = "Failed to create PSF from file: " + std::string(e.what());
        logMessage(error_msg);
        handleError(error_msg);
        throw std::runtime_error(error_msg);
    }
}


bool PSFGenerationService::isValidPSFType(const std::string& psf_type) const {
    auto it = std::find(supported_types_.begin(), supported_types_.end(), psf_type);
    return it != supported_types_.end();
}

std::string PSFGenerationService::saveResult(const std::string& path, const std::string& filename, std::shared_ptr<PSF> psf){
    // Use filesystem::path for better path handling
    std::filesystem::path base_path = path.empty() ? default_output_path_ : path;
    std::filesystem::path output_path = base_path / filename;  // Automatically handles separators
    
    // Ensure directory exists
    std::filesystem::create_directories(output_path.parent_path());
    
    std::string output_path_str = output_path.string();
    psf->saveAsTifFile(output_path_str);
    logMessage("PSF saved to: " + output_path_str);
    return output_path_str;
}

std::string PSFGenerationService::getExecutableDirectory() {
    try {
        // Get the path of the current executable
        std::filesystem::path execPath = std::filesystem::canonical("/proc/self/exe");
        return execPath.parent_path().string();
    } catch (const std::exception& e) {
        // Fallback to current working directory
        return std::filesystem::current_path().string();
    }
}