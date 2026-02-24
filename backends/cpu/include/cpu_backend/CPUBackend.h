#pragma once
#include "dolphinbackend/IBackend.h"
// #include "CPUBackendManager.h"

#include "dolphinbackend/Exceptions.h"
#include <fftw3.h>
#include <map>


class CPUBackendManager;
class FFTWManager;


struct CPUBackendConfig{
    bool useOMP = true;
    size_t ompThreads;
};

// Unified FFTW error check macro
#define FFTW_UNIFIED_CHECK(fftw_result, operation) { \
    if ((fftw_result) == nullptr) { \
        throw dolphin::backend::BackendException( \
            "FFTW operation failed", \
            "CPU", \
            operation \
        ); \
    } \
}
// Unified FFTW malloc error check macro
#define FFTW_MALLOC_UNIFIED_CHECK(ptr, size, operation) { \
    if ((ptr) == nullptr) { \
        throw dolphin::backend::MemoryException( \
            "FFTW memory allocation failed", \
            "CPU", \
            size, \
            operation \
        ); \
    } \
}



class CPUBackendMemoryManager : public IBackendMemoryManager{
public:
    CPUBackendMemoryManager(CPUBackendConfig config);
    
    ~CPUBackendMemoryManager();
    

    static size_t staticGetAvailableMemory();
    // Override device type method
    std::string getDeviceString() const noexcept override {
        return "cpu";
    }

    
    // Synchronization method - CPU implementation (no-op)
    void sync() override {}
    
    // Memory management initialization
    void setMemoryLimit(size_t maxMemorySize = 0) override;
    
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) const override;
    void allocateMemoryOnDevice(ComplexData& data) const override;
    ComplexData allocateMemoryOnDevice(const CuboidShape& shape) const override;
    bool isOnDevice(void* data) const override;
    ComplexData copyData(const ComplexData& srcdata) const override;
    ComplexData copyDataToDevice(const ComplexData& srcdata) const override; // for cpu these are copy operations
    ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const override; // for cpu these are copy operations
    void freeMemoryOnDevice(ComplexData& data) const override;
    size_t getAvailableMemory() const override;
    size_t getAllocatedMemory() const override;

    complex_t** createDataArray(std::vector<ComplexData*>& data) const override;


private:
    
    // Helper method to wait for memory availability
    void* allocateMemoryOnDevice(size_t) const;
    void waitForMemory(size_t requiredSize) const;
    static MemoryTracking cpuMemory; //static because currently only supports one device

};


class CPUDeconvolutionBackend : public IDeconvolutionBackend{
public:
    CPUDeconvolutionBackend(CPUBackendConfig config);
    ~CPUDeconvolutionBackend() override;
    
    // Override device type method
    std::string getDeviceString() const noexcept override {
        return "cpu";
    }

    // Synchronization method - CPU implementation (no-op)
    void sync() override {}

    // Core processing functions
    void cleanup() override;


    void initializePlan(const CuboidShape& cube) override;
     // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) const override;
    void backwardFFT(const ComplexData& in, ComplexData& out) const override;

    // Shift functions
    void octantFourierShift(ComplexData& data) const override;
    void inverseQuadrantShift(ComplexData& data) const override;

    // Complex arithmetic functions
    void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const override;
    void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexAddition(complex_t** data, ComplexData& sum, int nImages) const override;
    void sumToOneReal(complex_t** data, int nImages, int imageVolume) const override;
    void scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const override;
    void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const override;

    // Specialized functions
    void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const override;
    void normalizeImage(ComplexData& resultImage, real_t epsilon) const override;
    void rescaledInverse(ComplexData& data, real_t cubeVolume) const override;

    // Debug functions
    void hasNAN(const ComplexData& data) const override;

    // Layer and visualization functions
    void reorderLayers(ComplexData& data) const override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    void computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const override;

private:
    static FFTWManager fftwManager; 
    CPUBackendConfig config;
};



// Concrete CPU Backend Implementation
class CPUBackend : public IBackend {

    friend class CPUBackendManager;
private:

    static CPUBackend* create(CPUBackendConfig config) {
        auto deconv = std::make_unique<CPUDeconvolutionBackend>(config);
        auto memoryManager = std::make_unique<CPUBackendMemoryManager>(config);
        return new CPUBackend(std::move(deconv), std::move(memoryManager), config);
    }


    // Constructor for external ownership (references to externally-owned components)
    CPUBackend(CPUDeconvolutionBackend& deconv,
                            CPUBackendMemoryManager& mem,
                            CPUBackendConfig config)
            : deconvBackend(deconv),
                memoryManager(mem),
                owner(deconv, mem),
                config(config) {}

    // Constructor for self-ownership (takes ownership of both components)
    CPUBackend(std::unique_ptr<CPUDeconvolutionBackend> deconv,
                            std::unique_ptr<CPUBackendMemoryManager> mem,
                            CPUBackendConfig config)
            : deconvBackend(*deconv),
                memoryManager(*mem),
                owner(std::move(deconv), std::move(mem)),
                config(config) {}

    // Constructor for mixed ownership (takes ownership of deconv, external memory)
    CPUBackend(std::unique_ptr<CPUDeconvolutionBackend> deconv,
                            CPUBackendMemoryManager& mem,
                            CPUBackendConfig config)
            : deconvBackend(*deconv),
                memoryManager(mem),
                owner(std::move(deconv), mem),
                config(config) {}

    CPUDeconvolutionBackend& deconvBackend;
    CPUBackendMemoryManager& memoryManager;
    Owner owner;  // Always uses unique_ptr, nullptr for non-owned components
    CPUBackendConfig config;


    // Type-safe factory methods for different ownership models
    
    // Create CPUBackend with external ownership (references to externally-owned components)
    static std::shared_ptr<CPUBackend> createWithExternalOwnership(
        CPUDeconvolutionBackend& deconv,
        CPUBackendMemoryManager& mem,
        CPUBackendConfig config) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(deconv, mem, config));
    }

    // Create CPUBackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<CPUBackend> createWithSelfOwnership(
        std::unique_ptr<CPUDeconvolutionBackend> deconv,
        std::unique_ptr<CPUBackendMemoryManager> mem,
        CPUBackendConfig config) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(std::move(deconv), std::move(mem), config));
    }

    // Create CPUBackend with mixed ownership (takes ownership of deconv, external memory)
    static std::shared_ptr<CPUBackend> createWithMixedOwnership(
        std::unique_ptr<CPUDeconvolutionBackend> deconv,
        CPUBackendMemoryManager& mem,
        CPUBackendConfig config) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(std::move(deconv), mem, config));
    }

public:
    // Factory method to create CPUBackend with individual memory manager
    void sync() override {}

    CPUBackendConfig getConfig() {return config;}
    // Implementation of pure virtual methods
    std::string getDeviceString() const noexcept override {
        return "cpu";
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
            throw std::runtime_error("Cannot release deconvolution backend: not owned by this CPUBackend");
        }
        return owner.releaseDeconvBackend();
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() override {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this CPUBackend");
        }
        return owner.releaseMemoryManager();
    }

    // Take ownership of components
    void takeOwnership(std::unique_ptr<IDeconvolutionBackend> deconv) override {
        if (&(*deconv) != &deconvBackend) {
            throw std::runtime_error("Cannot take ownership: provided deconv backend is not the one currently referenced");
        }
        if (!owner.ownsDeconvBackend()) {
            throw std::runtime_error("Cannot take ownership: deconv backend is not owned by this CPUBackend");
        }
        owner.takeOwnership(std::move(deconv));
    }

    void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) override {
        if (&(*mem) != &memoryManager) {
            throw std::runtime_error("Cannot take ownership: provided memory manager is not the one currently referenced");
        }
        if (!owner.ownsMemoryManager()) {
            throw std::runtime_error("Cannot take ownership: memory manager is not owned by this CPUBackend");
        }
        owner.takeOwnership(std::move(mem));
    }

    // Memory manager access - for compatibility with shared ownership models
    std::shared_ptr<IBackendMemoryManager> getSharedMemoryManager() const noexcept override {
        if (ownsMemoryManager()) {
            // Return a shared_ptr that doesn't manage the lifetime (non-owning)
            return std::shared_ptr<IBackendMemoryManager>(&memoryManager, [](IBackendMemoryManager*){});
        } else {
            // For external ownership, we can't provide a proper shared_ptr
            return nullptr;
        }
    }

    // Direct pointer access
    IBackendMemoryManager* getMemoryManagerPtr() const noexcept override {
        return &memoryManager;
    }

    const IDeconvolutionBackend& getDeconvManager() const noexcept override {
        return deconvBackend;
    }

    const IBackendMemoryManager& getMemoryManager() const noexcept override {
        return memoryManager;
    }

    // Optionally, allow non-const access if you need modification
    IDeconvolutionBackend& mutableDeconvManager() noexcept override {
        return deconvBackend;
    }

    IBackendMemoryManager& mutableMemoryManager() noexcept override {
        return memoryManager;
    }


    
    // Overloaded version for CPU: simply return the original since CPU doesn't need complex_t thread management
    // IBackend& clone() override ;

    // IBackend& cloneSharedMemory() override;

    // void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads) override;
};