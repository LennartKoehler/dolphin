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
#include "dolphinbackend/Exceptions.h"
#include "dolphinbackend/IComputeBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include <cufft.h>
#include <CUBE.h>
#include <cuda_runtime.h>


class CUDABackendManager;



// Unified CUDA error check macro
#define CUDA_CHECK(err, operation) { \
    if (err == cudaErrorMemoryAllocation){ \
        throw dolphin::backend::MemoryException( \
            std::string("Temporary buffer allocation failed with CUDA error: ") \
            + cudaGetErrorString(err) + " (" + cudaGetErrorName(err) + ")", \
            "CUDA", \
            0, \
            operation \
        ); \
    } else if (err != cudaSuccess) { \
        throw dolphin::backend::BackendException( \
            std::string("CUDA error in '") + operation + "': " \
            + cudaGetErrorString(err) + " (" + cudaGetErrorName(err) + ")", \
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
            std::string("cuFFT error in '") + operation + "': code " + std::to_string(res), \
            "CUDA", \
            operation \
        ); \
    } \
}

#define CUFFT_RUNTIME_CHECK(call, operation) { \
    cufftResult res = call; \
    if (res != CUFFT_SUCCESS) { \
        throw dolphin::backend::BackendException( \
            "cuFFT error code: " + std::to_string(res), \
            "CUDA", \
            operation \
        ); \
    } \
}
struct cuFFTPlan{
    cufftHandle plan;
    FFTPlanDescription description;
};

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

    void sync() override {cudaStreamSynchronize(config.stream);}
    // Memory management initialization
    void setMemoryLimit(size_t maxMemorySize = 0) override;

    RealView reinterpret(ComplexData& data) const override;
    ComplexView reinterpret(RealData& data) const override;

    static size_t staticGetAvailableMemory();

    bool isOnDevice(const void* data) const override;
    size_t getAvailableMemory() const override;
    size_t getAllocatedMemory() const override;
    size_t estimateFFTWorkspace(const CuboidShape& shape) const override;

    void* copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const override;
    void* moveDataFromDevice(void* src, size_t size, const CuboidShape& shape,
                              const IBackendMemoryManager& destBackend) const override;
    void memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const override;
    void freeMemoryOnDevice(void* ptr, size_t size) const override;

private:

    // CUDA stream for memory operations
    CUDABackendConfig config;


    RealData allocateMemoryOnDeviceReal(const CuboidShape& shape) const override;
    RealData allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const override;
    ComplexData allocateMemoryOnDeviceComplex(const CuboidShape& shape) const override;
    ComplexData allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const override;
    void* allocateMemoryOnDevice(size_t requested_size) const override;

    // Method to get memory tracking instance
    MemoryTracking* getMemoryTracking() const { return config.device.memory; }
};

//these actually own the plan as the plan is streamspecific, and i should never have more than one of these on a stream
class CUDAComputeBackend : public virtual IComputeBackend{
public:
    explicit CUDAComputeBackend(CUDABackendConfig config);
    ~CUDAComputeBackend() override;

    // Override device type method
    std::string getDeviceString() const noexcept override {
        return (std::string("cuda") + std::to_string(config.device.id));
    }


    void initializePlan(const FFTPlanDescription& description) override;
    void sync() override {cudaStreamSynchronize(config.stream);}
    // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) const override;
    void backwardFFT(const ComplexData& in, ComplexData& out) const override;

    void forwardFFT(const RealData& in, ComplexData& out) const override;
    void backwardFFT(const ComplexData& in, RealData& out) const override;

    // Shift functions
    void octantFourierShift(ComplexData& data) const override;
    void octantFourierShift(RealData& data) const override;
    void inverseQuadrantShift(ComplexData& data) const override;

    // Complex arithmetic functions
    void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void multiplication(const RealData& a, const RealData& b, RealData& result) const override;
    void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const override;
    void division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const override;
    void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexAddition(complex_t** data, ComplexData& sum, int nImages, int imageVolume) const override;
    void sumToOne(real_t** data, int nImages, int imageVolume) const override;
    void sum(const ComplexData& data, complex_t* result) const override;
    void meanSquareError(const ComplexData& a, const ComplexData& b, real_t* result) const override;
    void scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const override;
    void scalarMultiplication(const RealData& a, real_t scalar, RealData& result) const override;
    void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const override;


    // void saveInterimImages(const ComplexData& resultImage, int gridNum, int channel_z, int i) const override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    // Gradient functions for real-valued data
    void gradientX(const RealData& image, RealData& gradX) const override;
    void gradientY(const RealData& image, RealData& gradY) const override;
    void gradientZ(const RealData& image, RealData& gradZ) const override;
    void gradient(const RealData& image, RealData& gradX, RealData& gradY, RealData& gradZ) const override;
    void divergence(const RealData& gx, const RealData& gy, const RealData& gz, RealData& result) const override;
    void divergence(const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& result) const override;
    void computeTV(real_t lambda, const ComplexData& div, ComplexData& tv) const override;
    // computeTV for real-valued divergence
    void computeTV(real_t lambda, const RealData& div, RealData& tv) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const override;
    // normalizeTV for real-valued gradients
    void normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t epsilon) const override;

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

    cufftHandle initializePlan_(const FFTPlanDescription& description);

    void createPlanComplex(cufftHandle& plan, const FFTPlanDescription& description) const;
    void createPlanRealToComplex(cufftHandle& plan, const FFTPlanDescription& description) const;
    void createPlanComplexToReal(cufftHandle& plan, const FFTPlanDescription& description) const ;

    void destroyPlans();

    cufftHandle getPlan(const FFTPlanDescription& description);
    void addPlan(const FFTPlanDescription& description, cufftHandle handle);
    std::vector<cuFFTPlan> cuFFTPlans;

    CUDABackendConfig config;

};




// Concrete CUDA Backend Implementation
class CUDABackend : public IBackend {
    friend CUDABackendManager;
private:


    // Constructor for external ownership (references to externally-owned components)
    CUDABackend(CUDABackendConfig config, CUDAComputeBackend& compute,
                CUDABackendMemoryManager& mem)
        : config(config),
          computeBackend(compute),
          memoryBackend(mem),
          owner(compute, mem) {}

    // Constructor for self-ownership (takes ownership of both components)
    CUDABackend(CUDABackendConfig config,
                std::unique_ptr<CUDAComputeBackend> compute,
                std::unique_ptr<CUDABackendMemoryManager> mem)
        : config(config),
          computeBackend(*compute),
          memoryBackend(*mem),
          owner(std::move(compute), std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of compute, external memory)
    CUDABackend(CUDABackendConfig config,
                std::unique_ptr<CUDAComputeBackend> compute,
                CUDABackendMemoryManager& mem)
        : config(config),
          computeBackend(*compute),
          memoryBackend(mem),
          owner(std::move(compute), mem) {}

    CUDAComputeBackend& computeBackend;
    CUDABackendMemoryManager& memoryBackend;
    Owner owner;  // Specialized CUDA owner
    CUDABackendConfig config;

    // Type-safe factory methods for different ownership models

    // Create CUDABackend with external ownership (references to externally-owned components)
    static std::shared_ptr<CUDABackend> createWithExternalOwnership(
        CUDABackendConfig config,
        CUDAComputeBackend& compute,
        CUDABackendMemoryManager& mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(config, compute, mem));
    }

    // Create CUDABackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<CUDABackend> createWithSelfOwnership(
        CUDABackendConfig config,
        std::unique_ptr<CUDAComputeBackend> compute,
        std::unique_ptr<CUDABackendMemoryManager> mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(config, std::move(compute), std::move(mem)));
    }

    // Create CUDABackend with mixed ownership (takes ownership of compute, external memory)
    static std::shared_ptr<CUDABackend> createWithMixedOwnership(
        CUDABackendConfig config,
        std::unique_ptr<CUDAComputeBackend> compute,
        CUDABackendMemoryManager& mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(config, std::move(compute), mem));
    }

public:
    // Factory method to create CUDABackend with

    static CUDABackend* create(CUDABackendConfig config) {
        try {
            auto compute = std::make_unique<CUDAComputeBackend>(config);
            auto memoryManager = std::make_unique<CUDABackendMemoryManager>(config);
            CUDABackend* backend = new CUDABackend(config, std::move(compute), std::move(memoryManager));

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
        return std::string("cuda") + std::to_string(static_cast<int>(config.device.id));
    }
    void sync() override{
        memoryBackend.sync();
    }

    const CUDABackendConfig& getConfig() const{
        return config;
    }

    // Ownership query methods
    bool ownsComputeBackend() const noexcept override {
        return owner.ownsComputeBackend();
    }

    bool ownsMemoryManager() const noexcept override {
        return owner.ownsMemoryManager();
    }

    // Memory manager is available if owner has it or if we have a reference
    bool hasMemoryManager() const noexcept override {
        return true; // We always have a reference to memory manager
    }

    // Ownership transfer methods for both components
    std::unique_ptr<IComputeBackend> releaseComputeBackend() override {
        if (!ownsComputeBackend()) {
            throw std::runtime_error("Cannot release compute backend: not owned by this CUDABackend");
        }
        return owner.releaseComputeBackend();
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() override {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this CUDABackend");
        }
        return owner.releaseMemoryManager();
    }

    // Take ownership of components
    void takeOwnership(std::unique_ptr<IComputeBackend> compute) override {
        if (&(*compute) != &computeBackend) {
            throw std::runtime_error("Cannot take ownership: provided compute backend is not the one currently referenced");
        }
        if (owner.ownsComputeBackend()) {
            throw std::runtime_error("Cannot take ownership: compute backend is already owned");
        }
        owner.takeOwnership(std::move(compute));
    }

    void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) override {
        if (&(*mem) != &memoryBackend) {
            throw std::runtime_error("Cannot take ownership: provided memory manager is not the one currently referenced");
        }
        if (owner.ownsMemoryManager()) {
            throw std::runtime_error("Cannot take ownership: memory manager is already owned");
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

    const IComputeBackend& getComputeManager() const noexcept override {
        return computeBackend;
    }

    const IBackendMemoryManager& getMemoryManager() const noexcept override {
        return memoryBackend;
    }

    // Optionally, allow non-const access if you need modification
    IComputeBackend& mutableComputeManager() noexcept override {
        return computeBackend;
    }

    IBackendMemoryManager& mutableMemoryManager() noexcept override {
        return memoryBackend;
    }
};
