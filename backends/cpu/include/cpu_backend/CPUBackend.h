#pragma once
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IComputeBackend.h"
// #include "CPUBackendManager.h"

#include <fftw3.h>


class CPUBackendManager;
class FFTWManager;



struct FFTWPlanDescription : public FFTPlanDescription{

    size_t ompThreads;

    FFTWPlanDescription(
        size_t ompThreads,
        PlanDirection direction,
        PlanType type,
        CuboidShape shape,
        bool inPlace
    ):
        ompThreads(ompThreads),
        FFTPlanDescription(direction, type, shape, inPlace){}

    bool operator==(const FFTWPlanDescription& other) const {
        return (FFTPlanDescription::operator==(other) && ompThreads == other.ompThreads);
    }
};

struct CPUBackendConfig{
    bool useOMP = true;
    size_t ompThreads;
};

// Unified FFTW error check macro
#define FFTW_UNIFIED_CHECK(fftw_result, operation) { \
    assert(fftw_result != nullptr);}
    // if ((fftw_result) == nullptr) { \
    //     throw dolphin::backend::BackendException( \
    //         "FFTW operation failed", \
    //         "CPU", \
    //         operation \
    //     ); \
    // } \

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
    CPUBackendMemoryManager(CPUBackendConfig config, MemoryTracking& memoryTracking);

    ~CPUBackendMemoryManager();


    static size_t staticGetAvailableMemory();
    // Override device type method
    std::string getDeviceString() const noexcept override {
        return "cpu";
    }

    DataView<real_t> reinterpret(ComplexData& data) const override;
    DataView<complex_t> reinterpret(RealData& data) const override;

    // Synchronization method - CPU implementation (no-op)
    void sync() override {}

    // Memory management initialization
    void setMemoryLimit(size_t maxMemorySize = 0) override;

    // Data management
    bool isOnDevice(const void* data) const override;
    size_t getAvailableMemory() const override;
    size_t getAllocatedMemory() const override;



    /**
     * Copy data from host to device
     * @param src Pointer to source data on host
     * @param size Size in bytes
     * @param shape Shape of the data
     * @return Pointer to allocated device memory
     */
    void* copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const override;


    RealData allocateMemoryOnDeviceReal(const CuboidShape& shape) const override;
    RealData allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const override;
    ComplexData allocateMemoryOnDeviceComplex(const CuboidShape& shape) const override;
    ComplexData allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const override;
    /**
     * Move data from device to another backend
     * @param src Pointer to source data on device
     * @param size Size in bytes
     * @param shape Shape of the data
     * @param destBackend Destination backend
     * @return Pointer to allocated memory on destination backend
     */
    void* moveDataFromDevice(void* src, size_t size, const CuboidShape& shape,
                                      const IBackendMemoryManager& destBackend) const override;
    /**
     * Memory copy between two pointers
     * @param src Pointer to source data
     * @param dest Pointer to destination data
     * @param size Size in bytes
     * @param shape Shape of the data
     */
    void memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const override;


    /**
     * Free memory on device
     * @param ptr Pointer to free
     * @param size Size in bytes (for tracking)
     */
    void freeMemoryOnDevice(void* ptr, size_t size) const override;

private:

    // Helper method to wait for memory availability
    void* allocateMemoryOnDevice(size_t size) const override;

    void waitForMemory(size_t requiredSize) const;
    MemoryTracking& memory; //static because currently only supports one device

};


class CPUComputeBackend : public virtual IComputeBackend{
public:
    CPUComputeBackend(CPUBackendConfig config, FFTWManager& manager);
    ~CPUComputeBackend() override;

    // Override device type method
    std::string getDeviceString() const noexcept override {
        return "cpu";
    }

    // Synchronization method - CPU implementation (no-op)
    void sync() override {}


    void initializePlan(const FFTPlanDescription& description) override ;
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

    // Specialized functions
    // void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const override;
    // void normalizeImage(ComplexData& resultImage, real_t epsilon) const override;
    // void rescaledInverse(ComplexData& data, real_t cubeVolume) const override;

    // Debug functions
    void hasNAN(const ComplexData& data) const override;


    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const override;
    // Gradient and TV functions for real-valued data
    void gradientX(const RealData& image, RealData& gradX) const override;
    void gradientY(const RealData& image, RealData& gradY) const override;
    void gradientZ(const RealData& image, RealData& gradZ) const override;
    void gradient(const RealData& image, RealData& gradX, RealData& gradY, RealData& gradZ) const override;
    void divergence(const RealData& gx, const RealData& gy, const RealData& gz, RealData& result) const override;
    void divergence(const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& result) const override;
    void computeTV(real_t lambda, const ComplexData& div, ComplexData& tv) const override;
    // computeTV for real-valued divergence
    void computeTV(real_t lambda, const RealData& div, RealData& tv) const override;
    void normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t epsilon) const override;

private:
    FFTWManager& fftwManager;
    CPUBackendConfig config;
};



// Concrete CPU Backend Implementation
class CPUBackend : public IBackend {

    friend class CPUBackendManager;

private:

    static CPUBackend* create(CPUBackendConfig config, FFTWManager& fftwManager, MemoryTracking& memory) {
        auto compute = std::make_unique<CPUComputeBackend>(config, fftwManager);
        auto memoryManager = std::make_unique<CPUBackendMemoryManager>(config, memory);
        return new CPUBackend(std::move(compute), std::move(memoryManager), config);
    }


    // Constructor for external ownership (references to externally-owned components)
    CPUBackend(CPUComputeBackend& compute,
                            CPUBackendMemoryManager& mem,
                            CPUBackendConfig config)
            : computeBackend(compute),
                memoryManager(mem),
                owner(compute, mem),
                config(config) {}

    // Constructor for self-ownership (takes ownership of both components)
    CPUBackend(std::unique_ptr<CPUComputeBackend> compute,
                            std::unique_ptr<CPUBackendMemoryManager> mem,
                            CPUBackendConfig config)
            : computeBackend(*compute),
                memoryManager(*mem),
                owner(std::move(compute), std::move(mem)),
                config(config) {}

    // Constructor for mixed ownership (takes ownership of compute, external memory)
    CPUBackend(std::unique_ptr<CPUComputeBackend> compute,
                            CPUBackendMemoryManager& mem,
                            CPUBackendConfig config)
            : computeBackend(*compute),
                memoryManager(mem),
                owner(std::move(compute), mem),
                config(config) {}

    CPUComputeBackend& computeBackend;
    CPUBackendMemoryManager& memoryManager;
    Owner owner;  // Always uses unique_ptr, nullptr for non-owned components
    CPUBackendConfig config;


    // Type-safe factory methods for different ownership models

    // Create CPUBackend with external ownership (references to externally-owned components)
    static std::shared_ptr<CPUBackend> createWithExternalOwnership(
        CPUComputeBackend& compute,
        CPUBackendMemoryManager& mem,
        CPUBackendConfig config) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(compute, mem, config));
    }

    // Create CPUBackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<CPUBackend> createWithSelfOwnership(
        std::unique_ptr<CPUComputeBackend> compute,
        std::unique_ptr<CPUBackendMemoryManager> mem,
        CPUBackendConfig config) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(std::move(compute), std::move(mem), config));
    }

    // Create CPUBackend with mixed ownership (takes ownership of compute, external memory)
    static std::shared_ptr<CPUBackend> createWithMixedOwnership(
        std::unique_ptr<CPUComputeBackend> compute,
        CPUBackendMemoryManager& mem,
        CPUBackendConfig config) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(std::move(compute), mem, config));
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
            throw std::runtime_error("Cannot release compute backend: not owned by this CPUBackend");
        }
        return owner.releaseComputeBackend();
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() override {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this CPUBackend");
        }
        return owner.releaseMemoryManager();
    }

    // Take ownership of components
    void takeOwnership(std::unique_ptr<IComputeBackend> compute) override {
        if (&(*compute) != &computeBackend) {
            throw std::runtime_error("Cannot take ownership: provided compute backend is not the one currently referenced");
        }
        if (!owner.ownsComputeBackend()) {
            throw std::runtime_error("Cannot take ownership: compute backend is not owned by this CPUBackend");
        }
        owner.takeOwnership(std::move(compute));
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

    const IComputeBackend& getComputeManager() const noexcept override {
        return computeBackend;
    }

    const IBackendMemoryManager& getMemoryManager() const noexcept override {
        return memoryManager;
    }

    // Optionally, allow non-const access if you need modification
    IComputeBackend& mutableComputeManager() noexcept override {
        return computeBackend;
    }

    IBackendMemoryManager& mutableMemoryManager() noexcept override {
        return memoryManager;
    }



    // Overloaded version for CPU: simply return the original since CPU doesn't need complex_t thread management
    // IBackend& clone() override ;

    // IBackend& cloneSharedMemory() override;

    // void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads) override;
};
