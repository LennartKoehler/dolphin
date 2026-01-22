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

#include "DeconvolutionService.h"
#include "PSFCreator.h"
#include "deconvolution/DeconvolutionProcessor.h"
#include "deconvolution/deconvolutionStrategies/IDeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategyPair.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include "DeconvolutionStrategyFactory.h"
#include "ThreadPool.h"
#include <chrono>
#include <fstream>
#include "IO/TiffReader.h"
#include "IO/TiffWriter.h"

DeconvolutionService::DeconvolutionService() 
    : initialized_(false),
      logger_([](const std::string& msg) { std::cout << "[DECONV_SERVICE] " << msg << std::endl; }),
      error_handler_([](const std::string& msg) { std::cerr << "[DECONV_ERROR] " << msg << std::endl; }),
      thread_pool_(std::make_unique<ThreadPool>(1)){}

DeconvolutionService::~DeconvolutionService() {
    shutdown();
}




void DeconvolutionService::initialize() {
    if (initialized_) return;
    
    try {
        DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
        supported_algorithms_ = fact.getAvailableAlgorithms();

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
        
        initialized_ = true;
        logMessage("Deconvolution Service initialized successfully");
    } catch (const std::exception& e) {
        handleError("Failed to initialize Deconvolution Service: " + std::string(e.what()));
        initialized_ = false;
        throw;
    }
}

bool DeconvolutionService::isInitialized() const {
    return initialized_;
}

void DeconvolutionService::shutdown() {
    if (!initialized_) return;
    
    initialized_ = false;
    logMessage("Deconvolution Service shut down successfully");
}

std::unique_ptr<DeconvolutionResult> DeconvolutionService::deconvolve(const DeconvolutionRequest& request) {
    auto start_time = std::chrono::high_resolution_clock::now();


    try {
        if (!initialized_) {
            throw std::runtime_error("Deconvolution Service not initialized");
        }
        
        logMessage("Starting deconvolution with request: " + request.getConfig()->imagePath);

        // Validate request
        if (!validateDeconvolutionRequest(request)) {
            std::string error_msg = "Invalid deconvolution request";
            handleError(error_msg);
            return createResult(false, error_msg, 
                              std::chrono::duration<double>::zero());
        }

        // unpack
        std::shared_ptr<SetupConfig> setupConfig = request.getConfig();
        std::shared_ptr<DeconvolutionConfig> deconvConfig = setupConfig->deconvolutionConfig;

        // Load hyperstack
        // std::shared_ptr<Hyperstack> hyperstack = loadImage(setupConfig->imagePath);
        // if (!hyperstack) {
        //     std::string error_msg = "Failed to load image from: " + setupConfig->imagePath;
        //     handleError(error_msg);
        //     return createResult(false, error_msg,
        //                       std::chrono::duration<double>::zero());
        // }

        // Create PSFs
        std::vector<PSF> psfs = createPSFsFromSetup(setupConfig);
        if (request.getPSFConfig() != nullptr){
            psfs.push_back(PSFCreator::generatePSFFromPSFConfig(request.getPSFConfig(), thread_pool_.get()));
        }
        if (psfs.empty()) {
            std::string error_msg = "No valid PSFs provided";
            handleError(error_msg);
            return createResult(false, error_msg,
                              std::chrono::duration<double>::zero());
        }
        
        // Create deconvolution strategy pair using factory
        DeconvolutionStrategyFactory& factory = DeconvolutionStrategyFactory::getInstance();
        auto strategyPair = factory.createStrategyPair(setupConfig);
        
        if (!strategyPair) {
            std::string error_msg = "Unsupported deconvolution type: " + deconvConfig->deconvolutionType;
            handleError(error_msg);
            return createResult(false, error_msg,
                              std::chrono::duration<double>::zero());
        }
        
        // Configure both strategy and executor
        strategyPair->getExecutor().configure(std::make_unique<DeconvolutionConfig>(*deconvConfig.get()));

        // Execute the plan using the executor
        std::string output_path = request.output_path;

        std::string path = output_path + "/deconv_" + TiffReader::getFilename(setupConfig->imagePath);
        // TiffWriter writer{output_path + "/deconv_" + UtlIO::getFilename(setupConfig->imagePath)};
        
        int channel = 0;
        std::shared_ptr<TiffReader> reader = std::make_shared<TiffReader>(setupConfig->imagePath, channel);
        std::shared_ptr<TiffWriter> writer = std::make_shared<TiffWriter>(path, reader->getMetaData().getShape());
       
        DeconvolutionPlan plan = strategyPair->getStrategy().createPlan(reader, writer, psfs, *deconvConfig, *setupConfig);


        strategyPair->getExecutor().execute(plan);
        
        // Load the result from the writer
        // Hyperstack result = loadImage(output_path + "/deconv_" + UtlIO::getFilename(setupConfig->imagePath));
        
        // Handle saving
        std::vector<std::string> layer_paths;
        
        // if (!output_path.empty()) {
        //     std::string path = output_path + "/deconv_" + UtlIO::getFilename(setupConfig->imagePath);
        //     result.saveAsTifFile(path);
        //     logMessage("Deconvolution result saved to: " + path);
            
        //     if (request.save_separate) {
        //         result.saveAsTifDir("../result/deconv");
        //     }
        // }
        
        if (request.show_example) {
            logMessage("Showing example deconvolution layer visualization");
            // result.showChannel(0); // Uncomment when available
        }
        
        if (request.print_info) {
            logMessage("Image metadata:");
            // hyperstack->printMetadata(); // Uncomment when available
        }
        
        // Create result
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        auto result_obj = createResult(true, "Deconvolution completed successfully", duration);
        result_obj->output_path = output_path;
        result_obj->individual_layer_paths = layer_paths;
        
        return result_obj;
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::string error_msg = "Deconvolution failed: " + std::string(e.what());
        logMessage(error_msg);
        return createResult(false, error_msg, duration);
    }
}

std::future<std::unique_ptr<DeconvolutionResult>> DeconvolutionService::deconvolveAsync(
    const DeconvolutionRequest& request){
        return thread_pool_->enqueue([this, request]() {
            return deconvolve(request);
        });
    }

std::future<std::vector<std::unique_ptr<DeconvolutionResult>>> DeconvolutionService::deconvolveBatchAsync(
    const std::vector<DeconvolutionRequest>& requests){
    
    return thread_pool_->enqueue([this, requests]() {
        std::vector<std::unique_ptr<DeconvolutionResult>> results;
        results.reserve(requests.size());
        
        for (size_t i = 0; i < requests.size(); ++i) {
            // Update progress
            if (progress_callback_) {
                std::lock_guard<std::mutex> lock(progress_mutex_);
                progress_callback_(static_cast<int>((i * 100) / requests.size()));
            }
            
            results.push_back(deconvolve(requests[i]));
        }
        
        return results;
    });
}


std::vector<std::string> DeconvolutionService::getSupportedAlgorithms() const {
    return supported_algorithms_;
}

std::vector<std::string> DeconvolutionService::getSupportedStrategyTypes() const {
    DeconvolutionStrategyFactory& factory = DeconvolutionStrategyFactory::getInstance();
    return factory.getSupportedTypes();
}

bool DeconvolutionService::validateAlgorithmConfig(const std::string& algorithm, const json& config) const {
    auto it = std::find(supported_algorithms_.begin(), supported_algorithms_.end(), algorithm);
    if (it == supported_algorithms_.end()) {
        return false;
    }
    
    // Add algorithm-specific validation here
    return true;
}

void DeconvolutionService::setLogger(std::function<void(const std::string&)> logger) {
    logger_ = logger ? logger : default_logger_;
}

void DeconvolutionService::setConfigLoader(std::function<json(const std::string&)> loader) {
    config_loader_ = loader;
}



void DeconvolutionService::setDefaultLogger(std::function<void(const std::string&)> logger) {
    default_logger_ = logger;
    if (!logger_) {
        logger_ = default_logger_;
    }
}

void DeconvolutionService::setErrorHandler(std::function<void(const std::string&)> handler) {
    error_handler_ = handler ? handler : default_error_handler_;
}

std::unique_ptr<DeconvolutionResult> DeconvolutionService::createResult(
    bool success,
    const std::string& message,
    std::chrono::duration<double> duration) {
    
    auto result = std::make_unique<DeconvolutionResult>(success, message, duration);
    return result;
}

void DeconvolutionService::logMessage(const std::string& message) {
    if (logger_) {
        logger_("INFO: " + message);
    }
}

void DeconvolutionService::handleError(const std::string& error) {
    if (error_handler_) {
        error_handler_("ERROR: " + error);
    }
}

bool DeconvolutionService::validateDeconvolutionRequest(const DeconvolutionRequest& request) const {
    if (request.getConfig()->imagePath.empty()) {
        return false;
    }
    
    auto algorithm_it = std::find(
        supported_algorithms_.begin(), 
        supported_algorithms_.end(), 
        request.getConfig()->deconvolutionConfig->algorithmName
    );
    
    if (algorithm_it == supported_algorithms_.end()) {
        logger_("ERROR: could not find algorithm");
        return false;
    }
    
    return true;
}



std::vector<PSF> DeconvolutionService::createPSFsFromSetup(
    std::shared_ptr<SetupConfig> setupConfig) {

    std::vector<PSF> psfs;

    if (!setupConfig->psfConfigPath.empty()){
        std::shared_ptr<PSFConfig> config = PSFCreator::generatePSFConfigFromConfigPath(setupConfig->psfConfigPath);
        psfs.push_back(PSFCreator::generatePSFFromPSFConfig(config, thread_pool_.get()));
    }
    if (!setupConfig->psfFilePath.empty()){
        std::vector<PSF> filePsfs = PSFCreator::readPSFsFromFilePath(setupConfig->psfFilePath);
        psfs.insert(psfs.end(), filePsfs.begin(), filePsfs.end());
    }
    if (!setupConfig->psfDirPath.empty()){
        std::vector<std::shared_ptr<PSFConfig>> psfconfigs = PSFCreator::generatePSFsFromDir(setupConfig->psfDirPath);
        for (auto psfconfig : psfconfigs){
            psfs.push_back(PSFCreator::generatePSFFromPSFConfig(psfconfig, thread_pool_.get()));
        }
    }
    return psfs;
}



void DeconvolutionService::setProgressCallback(std::function<void(int)> callback){
    this->progress_callback_ = callback;
}