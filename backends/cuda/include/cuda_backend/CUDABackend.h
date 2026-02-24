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

#pragma once
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IDeconvolutionBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include "dolphinbackend/Exceptions.h"
#include <cufft.h>
#include <CUBE.h>
#include <cuda_runtime.h>
#include <map>
#include <iostream>


class CUDABackendManager;

// Unified CUDA error check macro
#define CUDA_CHECK(err, operation) { \
    if (err == cudaErrorMemoryAllocation){ \
        throw dolphin::backend::MemoryException( \
            "Temoprary buffer allocation failed with " + std::string("CUDA error: ") + cudaGetErrorString(err), \
            "CUDA", \
            0, \
            operation \
        ); \
    } \
    else if (err != cudaSuccess) { \
        throw dolphin::backend::BackendException( \
            std::string("CUDA error: ") + cudaGetErrorString(err), \
            "CUDA", \
            operation \
        ); \
    } \
}
#define CUDA_MEMORY_ALLOC_CHECK(err, size, operation) { \
    if (err != cudaSuccess){ \
        throw dolphin::backend::MemoryException( \
            "Memory allocation failed with " + std::string("CUDA error: ") + cudaGetErrorString(err), \
            "CUDA", \
            size, \
            operation \
        ); \
    } \
}

// Unified cuFFT error check macro
#define CUFFT_CHECK(call, operation) { \
    cufftResult res = call; \
    if (res != CUFFT_SUCCESS) { \
        throw dolphin::backend::BackendException( \
            "cuFFT error code: " + std::to_string(res), \
            "CUDA", \
            operation \
        ); \
    } \
}



using cudaDeviceID = int;
struct CUDADevice{
    cudaDeviceID id = 0;
    MemoryTracking* memory;
};

struct CUDABackendConfig{
    CUDADevice device;
    cudaStream_t stream = cudaStreamLegacy;
};


class CUDABackendMemoryManager : public IBackendMemoryManager{
public:
    // Constructor
    explicit CUDABackendMemoryManager(CUDABackendConfig config);
    ~CUDABackendMemoryManager();
    
    // Override device type method
    std::string getDeviceString() const noexcept override {
        return std::string("cuda") + std::to_string(config.device.id);
    }
    
    void sync() override {cudaStreamSynchronize(stream);}
    // Memory management initialization
    void setMemoryLimit(size_t maxMemorySize = 0) override;
    
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) const override;
    void allocateMemoryOnDevice(ComplexData& data) const override;
    ComplexData allocateMemoryOnDevice(const CuboidShape& shape) const override;
    bool isOnDevice(void* data) const override;
    ComplexData copyData(const ComplexData& srcdata) const override;
    ComplexData copyDataToDevice(const ComplexData& srcdata) const override; // for gpu these are copy operations
    ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const override; // for gpu these are copy operations

    complex_t** createDataArray(std::vector<ComplexData*>& data) const override ;
    void freeMemoryOnDevice(ComplexData& data) const override;
    size_t getAvailableMemory() const override;
    size_t getAllocatedMemory() const override;
private:

    // CUDA stream for memory operations
    CUDABackendConfig config; 
    // Helper method to wait for memory availability
    void waitForMemory(size_t requiredSize) const;

    void* allocateMemoryOnDevice(size_t requested_size) const ;
    
    // Static method to get memory tracking instance
    MemoryTracking* getMemoryTracking() { return config.device.memory; }
};

//these actually own the plan as the plan is streamspecific, and i should never have more than one of these on a stream
class CUDADeconvolutionBackend : public IDeconvolutionBackend{
public:
    explicit CUDADeconvolutionBackend(CUDABackendConfig config);
    ~CUDADeconvolutionBackend() override;
    
    // Override device type method
    std::string getDeviceString() const noexcept override {
        return (std::string("cuda") + std::to_string(device.id));
    }


    void sync() override {cudaStreamSynchronize(stream);}
    // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) const override;
    void backwardFFT(const ComplexData& in, ComplexData& out) const override;

    // Shift functions
    void octantFourierShift(ComplexData& data) const override;

    // Complex arithmetic functions
    void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const override;
    void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexAddition(complex_t** data, ComplexData& sum, int nImages) const override;
    void scalarMultiplication(const ComplexData& a, complex_t  scalar, ComplexData& result) const override;
    void sumToOneReal(complex_t** data, int nImages, int imageVolume) const override;
    void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const override;

    // Specialized functions
    void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const override;

    // void saveInterimImages(const ComplexData& resultImage, int gridNum, int channel_z, int i) const override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    void computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const override;

    // Layer and visualization functions
    // void reorderLayers(ComplexData& data) override;
    // void visualizeFFT(const ComplexData& data) override;

    // Conversion functions
    // void readCVMat(const std::vector<cv::Mat>& input, ComplexData& output) override;
    // void convertFFTWComplexToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
    // void convertFFTWComplexRealToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
    // void convertFFTWComplexImgToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
   void hasNAN(const ComplexData& data) const override;




private:
    
    void initializePlan(const CuboidShape& shape); 
    cufftHandle forward = 0;
    cufftHandle backward = 0;
    CuboidShape planSize;
    CUDABackendConfig config;

};




// Concrete CUDA Backend Implementation
class CUDABackend : public IBackend {
    friend CUDABackendManager;
private:


    // Constructor for external ownership (references to externally-owned components)
    CUDABackend(CUDABackendConfig config, CUDADeconvolutionBackend& deconv,
                CUDABackendMemoryManager& mem)
        : config(config),
          deconvBackend(deconv),
          memoryBackend(mem),
          owner(deconv, mem) {}

    // Constructor for self-ownership (takes ownership of both components)
    CUDABackend(CUDABackendConfig config,
                std::unique_ptr<CUDADeconvolutionBackend> deconv,
                std::unique_ptr<CUDABackendMemoryManager> mem)
        : config(config),
          deconvBackend(*deconv),
          memoryBackend(*mem),
          owner(std::move(deconv), std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of deconv, external memory)
    CUDABackend(CUDABackendConfig config,
                std::unique_ptr<CUDADeconvolutionBackend> deconv,
                CUDABackendMemoryManager& mem)
        : config(config),
          deconvBackend(*deconv),
          memoryBackend(mem),
          owner(std::move(deconv), mem) {}

    CUDADeconvolutionBackend& deconvBackend;
    CUDABackendMemoryManager& memoryBackend;
    Owner owner;  // Specialized CUDA owner
    CUDABackendConfig config;

    // Type-safe factory methods for different ownership models
    
    // Create CUDABackend with external ownership (references to externally-owned components)
    static std::shared_ptr<CUDABackend> createWithExternalOwnership(
        CUDABackendConfig config,
        CUDADeconvolutionBackend& deconv,
        CUDABackendMemoryManager& mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(config, deconv, mem));
    }

    // Create CUDABackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<CUDABackend> createWithSelfOwnership(
        CUDABackendConfig config,
        std::unique_ptr<CUDADeconvolutionBackend> deconv,
        std::unique_ptr<CUDABackendMemoryManager> mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(config, std::move(deconv), std::move(mem)));
    }

    // Create CUDABackend with mixed ownership (takes ownership of deconv, external memory)
    static std::shared_ptr<CUDABackend> createWithMixedOwnership(
        CUDABackendConfig config,
        std::unique_ptr<CUDADeconvolutionBackend> deconv,
        CUDABackendMemoryManager& mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(config, std::move(deconv), mem));
    }

public:
    // Factory method to create CUDABackend with
    
    static CUDABackend* create(CUDABackendConfig config) {
        try {
            auto deconv = std::make_unique<CUDADeconvolutionBackend>(config);
            auto memoryManager = std::make_unique<CUDABackendMemoryManager>(config);
            CUDABackend* backend = new CUDABackend(config, std::move(deconv), std::move(memoryManager));
            
            // size_t freeMem, totalMem;
            // cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
            // CUDA_CHECK(err, "create - cudaMemGetInfo");
            
            // if (totalMem == 0) {
            //     throw dolphin::backend::BackendException(
            //         "Device 0 reports zero memory", "CUDA", "create");
            // }
            
            // backend->setDevice(CUDADevice{0, new MemoryTracking(totalMem)});
            return backend;
        } catch (...) {
            // Clean up any allocated resources if creation fails
            throw dolphin::backend::BackendException(
                "Failed to create CUDABackend", "CUDA", "create");
        }
    }


    // Implementation of pure virtual methods
    std::string getDeviceString() const noexcept override {
        return std::string("cuda") + std::to_string(static_cast<int>(device.id));
    }
    void sync() override{
        memoryBackend.sync();
    }

    const CUDABackendConfig& getConfig() const{
        return config;
    }

    // Ownership query methods
    bool ownsDeconvolutionBackend() const noexcept override {
        return owner.ownsDeconvBackend();
    }

    bool ownsMemoryManager() const noexcept override {
        return owner.ownsMemoryManager();
    }

    // Memory manager is available if owner has it or if we have a reference
    bool hasMemoryManager() const noexcept override {
        return true; // We always have a reference to memory manager
    }

    // Ownership transfer methods for both components
    std::unique_ptr<IDeconvolutionBackend> releaseDeconvolutionBackend() override {
        if (!ownsDeconvolutionBackend()) {
            throw std::runtime_error("Cannot release deconvolution backend: not owned by this CUDABackend");
        }
        return owner.releaseDeconvBackend();
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() override {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this CUDABackend");
        }
        return owner.releaseMemoryManager();
    }

    // Take ownership of components
    void takeOwnership(std::unique_ptr<IDeconvolutionBackend> deconv) override {
        if (&(*deconv) != &deconvBackend) {
            throw std::runtime_error("Cannot take ownership: provided deconv backend is not the one currently referenced");
        }
        if (!owner.ownsDeconvBackend()) {
            throw std::runtime_error("Cannot take ownership: deconv backend is not owned by this CUDABackend");
        }
        owner.takeOwnership(std::move(deconv));
    }

    void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) override {
        if (&(*mem) != &memoryBackend) {
            throw std::runtime_error("Cannot take ownership: provided memory manager is not the one currently referenced");
        }
        if (!owner.ownsMemoryManager()) {
            throw std::runtime_error("Cannot take ownership: memory manager is not owned by this CUDABackend");
        }
        owner.takeOwnership(std::move(mem));
    }

    // Memory manager access - for compatibility with shared ownership models
    std::shared_ptr<IBackendMemoryManager> getSharedMemoryManager() const noexcept override {
        if (ownsMemoryManager()) {
            // Return a shared_ptr that doesn't manage the lifetime (non-owning)
            return std::shared_ptr<IBackendMemoryManager>(&memoryBackend, [](IBackendMemoryManager*){});
        } else {
            // For external ownership, we can't provide a proper shared_ptr
            return nullptr;
        }
    }

    // Direct pointer access
    IBackendMemoryManager* getMemoryManagerPtr() const noexcept override {
        return &memoryBackend;
    }

    const IDeconvolutionBackend& getDeconvManager() const noexcept override {
        return deconvBackend;
    }

    const IBackendMemoryManager& getMemoryManager() const noexcept override {
        return memoryBackend;
    }

    // Optionally, allow non-const access if you need modification
    IDeconvolutionBackend& mutableDeconvManager() noexcept override {
        return deconvBackend;
    }

    IBackendMemoryManager& mutableMemoryManager() noexcept override {
        return memoryBackend;
    }
};