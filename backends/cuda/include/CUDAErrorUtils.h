#pragma once
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include "dolphinbackend/Exceptions.h"

namespace dolphin {
namespace backend {
namespace cuda_utils {

// Enhanced CUDA error checking with detailed context
inline void checkCUDAError(cudaError_t err, const std::string& operation, 
                          const std::string& context = "", int line = 0) {
    if (err != cudaSuccess) {
        std::string detailed_msg = "CUDA error: " + std::string(cudaGetErrorString(err));
        
        // Add operation context
        detailed_msg += " | Operation: " + operation;
        
        // Add additional context if provided
        if (!context.empty()) {
            detailed_msg += " | Context: " + context;
        }
        
        // Add line number if available
        if (line > 0) {
            detailed_msg += " | Line: " + std::to_string(line);
        }
        
        // Print to stderr for immediate visibility
        std::cerr << "[CUDA ERROR] " << detailed_msg << std::endl;
        
        // Also print current device info
        int current_device;
        cudaError_t get_device_err = cudaGetDevice(&current_device);
        if (get_device_err == cudaSuccess) {
            std::cerr << "[CUDA INFO] Current device: " << current_device << std::endl;
            
            // Get device properties
            cudaDeviceProp props;
            cudaError_t prop_err = cudaGetDeviceProperties(&props, current_device);
            if (prop_err == cudaSuccess) {
                std::cerr << "[CUDA INFO] Device: " << props.name << std::endl;
                std::cerr << "[CUDA INFO] Memory: " << props.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
            }
        }
        
        // Check for any pending errors
        cudaError_t last_error = cudaGetLastError();
        if (last_error != cudaSuccess && last_error != err) {
            std::cerr << "[CUDA ERROR] Additional pending error: " << cudaGetErrorString(last_error) << std::endl;
        }
        
        throw BackendException(detailed_msg, "CUDA", operation);
    }
}

// Enhanced cuFFT error checking
inline void checkCUFFTError(cufftResult res, const std::string& operation, 
                           const std::string& context = "", int line = 0) {
    if (res != CUFFT_SUCCESS) {
        std::string detailed_msg = "cuFFT error code: " + std::to_string(res);
        
        // Add operation context
        detailed_msg += " | Operation: " + operation;
        
        // Add additional context if provided
        if (!context.empty()) {
            detailed_msg += " | Context: " + context;
        }
        
        // Add line number if available
        if (line > 0) {
            detailed_msg += " | Line: " + std::to_string(line);
        }
        
        // Print to stderr for immediate visibility
        std::cerr << "[CUFFT ERROR] " << detailed_msg << std::endl;
        
        // Add error description based on cuFFT error code
        std::string error_desc;
        switch (res) {
            case CUFFT_INVALID_PLAN:
                error_desc = "Invalid plan";
                break;
            case CUFFT_ALLOC_FAILED:
                error_desc = "Allocation failed";
                break;
            case CUFFT_INVALID_TYPE:
                error_desc = "Invalid data type";
                break;
            case CUFFT_INVALID_VALUE:
                error_desc = "Invalid value";
                break;
            case CUFFT_INTERNAL_ERROR:
                error_desc = "Internal error";
                break;
            case CUFFT_EXEC_FAILED:
                error_desc = "Execution failed";
                break;
            case CUFFT_SETUP_FAILED:
                error_desc = "Setup failed";
                break;
            case CUFFT_INVALID_SIZE:
                error_desc = "Invalid size";
                break;
            case CUFFT_UNALIGNED_DATA:
                error_desc = "Unaligned data";
                break;
            case CUFFT_INCOMPLETE_PARAMETER_LIST:
                error_desc = "Incomplete parameter list";
                break;
            case CUFFT_INVALID_DEVICE:
                error_desc = "Invalid device";
                break;
            case CUFFT_PARSE_ERROR:
                error_desc = "Parse error";
                break;
            case CUFFT_NO_WORKSPACE:
                error_desc = "No workspace";
                break;
            case CUFFT_NOT_IMPLEMENTED:
                error_desc = "Not implemented";
                break;
            case CUFFT_LICENSE_ERROR:
                error_desc = "License error";
                break;
            case CUFFT_NOT_SUPPORTED:
                error_desc = "Not supported";
                break;
            default:
                error_desc = "Unknown error";
                break;
        }
        
        std::cerr << "[CUFFT INFO] Error description: " << error_desc << std::endl;
        
        throw BackendException(detailed_msg, "CUDA", operation);
    }
}

// Macro versions for easier use
#define CUDA_CHECK_EX(err, operation, context) \
    dolphin::backend::cuda_utils::checkCUDAError(err, operation, context, __LINE__)

#define CUFFT_CHECK_EX(res, operation, context) \
    dolphin::backend::cuda_utils::checkCUFFTError(res, operation, context, __LINE__)

// Original macros for backward compatibility
#define CUDA_CHECK(err, operation) \
    dolphin::backend::cuda_utils::checkCUDAError(err, operation, "", __LINE__)

#define CUFFT_CHECK(call, operation) \
    dolphin::backend::cuda_utils::checkCUFFTError(call, operation, "", __LINE__)

// Memory debugging utilities
inline void printMemoryInfo(const std::string& context = "") {
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    
    if (err == cudaSuccess) {
        std::cerr << "[CUDA MEMORY] " << context << " - Free: " 
                  << (freeMem / 1024 / 1024) << " MB, Total: " 
                  << (totalMem / 1024 / 1024) << " MB" << std::endl;
    } else {
        std::cerr << "[CUDA MEMORY] Failed to get memory info: " 
                  << cudaGetErrorString(err) << std::endl;
    }
}

// Check for CUDA errors without throwing (for debugging)
inline bool checkCUDAErrorSilent() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR SILENT] " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

} // namespace cuda_utils
} // namespace backend
} // namespace dolphin