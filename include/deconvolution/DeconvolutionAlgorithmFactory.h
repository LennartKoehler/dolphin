#pragma once

#include <memory>
#include <utility>

#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"

#ifdef CUDA_AVAILABLE
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#endif

/**
 * @brief Factory class for creating deconvolution algorithm instances with CPU/GPU variants.
 * 
 * This factory supports both CPU and GPU variants of algorithms, with conditional 
 * compilation to ensure GPU variants are only registered when CUDA is available.
 */
class DeconvolutionAlgorithmFactory {
public:
    using AlgorithmCreator = std::function<std::shared_ptr<BaseDeconvolutionAlgorithm>()>;

    static DeconvolutionAlgorithmFactory& getInstance() {
        static DeconvolutionAlgorithmFactory instance;
        return instance;
    }

    void registerAlgorithm(const std::string& name, AlgorithmCreator creator) {
        algorithms_[name] = creator;
    }

    /**
     * @brief Create an algorithm instance based on configuration.
     * @param config Deconvolution configuration containing algorithm selection
     * @return Shared pointer to the created algorithm instance
     * @throws std::runtime_error if algorithm is unknown or GPU variant requested but unavailable
     */
    std::shared_ptr<BaseDeconvolutionAlgorithm> create(
        const DeconvolutionConfig& config
    ) {
        auto it = algorithms_.find(config.algorithmName);
        if (it == algorithms_.end()) {
            throw std::runtime_error("Unknown algorithm: " + config.algorithmName);
        }
        
        // Check if GPU variant is requested but CUDA is not available
        if (isGPUVariant(config.algorithmName) && !isGPUSupported()) {
            throw std::runtime_error("GPU variant '" + config.algorithmName + 
                                   "' requested but CUDA is not available");
        }
        
        auto algorithm = it->second();
        algorithm->configure(config);
        return algorithm;
    }

    /**
     * @brief Get list of all available algorithms.
     * @return Vector of algorithm names
     */
    std::vector<std::string> getAvailableAlgorithms() const {
        std::vector<std::string> names;
        for (const auto& pair : algorithms_) {
            names.push_back(pair.first);
        }
        return names;
    }

    /**
     * @brief Get list of GPU-available algorithms.
     * @return Vector of GPU algorithm names (empty if CUDA not available)
     */
    std::vector<std::string> getGPUAlgorithms() const {
        std::vector<std::string> gpu_names;
        for (const auto& pair : algorithms_) {
            if (isGPUVariant(pair.first) && isGPUSupported()) {
                gpu_names.push_back(pair.first);
            }
        }
        return gpu_names;
    }

    /**
     * @brief Check if a specific algorithm name represents a GPU variant.
     * @param name Algorithm name to check
     * @return true if GPU variant, false if CPU variant
     */
    bool isGPUVariant(const std::string& name) const {
        return name.find("GPU") != std::string::npos || 
               name.find("gpu") != std::string::npos ||
               name.find("Cuda") != std::string::npos ||
               name.find("cuda") != std::string::npos;
    }

    /**
     * @brief Check if CUDA/GPU support is available.
     * @return true if CUDA is available, false otherwise
     */
    bool isGPUSupported() const {
        return is_cuda_available_;
    }

private:
    DeconvolutionAlgorithmFactory()
        : is_cuda_available_(false) {
        
        // Detect CUDA availability
        is_cuda_available_ = detectCUDAAvailability();
        
        // Register all CPU algorithms
        registerCPUAlgorithms();
        
        // Register GPU algorithms if CUDA is available
        if (is_cuda_available_) {
            registerGPUAlgorithms();
        }
    }

    bool detectCUDAAvailability() {
        #ifdef CUDA_AVAILABLE
        try {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err == cudaSuccess && device_count > 0) {
                return true;
            }
        } catch (...) {
            // CUDA not available
        }
        #endif
        return false;
    }

    void registerCPUAlgorithms() {
        registerAlgorithm("InverseFilter", []() {
            return std::make_unique<InverseFilterDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucy", []() {
            return std::make_unique<RLDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucyTotalVariation", []() {
            return std::make_unique<RLTVDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RegularizedInverseFilter", []() {
            return std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucywithAdaptiveDamping", []() {
            return std::make_unique<RLADDeconvolutionAlgorithm>();
        });
    }

#ifdef CUDA_AVAILABLE
    void registerGPUAlgorithms() {
        // Register GPU variants with conditional compilation
        registerAlgorithm("InverseFilterGPU", []() {
            return std::make_unique<InverseFilterDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucyGPU", []() {
            return std::make_unique<RLDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucyTotalVariationGPU", []() {
            return std::make_unique<RLTVDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RegularizedInverseFilterGPU", []() {
            return std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
        });
        registerAlgorithm("RichardsonLucywithAdaptiveDampingGPU", []() {
            return std::make_unique<RLADDeconvolutionAlgorithm>();
        });
        
        std::cout << "[INFO] GPU algorithm variants registered successfully" << std::endl;
    }
#else
    void registerGPUAlgorithms() {
        // No GPU algorithms to register when CUDA is not available
    }
#endif

    std::unordered_map<std::string, AlgorithmCreator> algorithms_;
    bool is_cuda_available_;
};
