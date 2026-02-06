#pragma once
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IDeconvolutionBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include <fftw3.h>
#include <map>




extern void set_backend_logger(LogCallback cb);


class CPUBackendMemoryManager : public IBackendMemoryManager{
public:
    CPUBackendMemoryManager();
    
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


private:
    // Memory management
    static MemoryTracking memory; // only implemented  cpu for a single cpu device
    
    // Helper method to wait for memory availability
    void* allocateMemoryOnDevice(size_t) const;
    void waitForMemory(size_t requiredSize) const;
          // Static method to get memory tracking instance
    static MemoryTracking& getMemoryTracking() { return memory; }

};


class CPUDeconvolutionBackend : public IDeconvolutionBackend{
public:
    CPUDeconvolutionBackend();
    ~CPUDeconvolutionBackend() override;
    
    // Override device type method
    std::string getDeviceString() const noexcept override {
        return "cpu";
    }

    // Synchronization method - CPU implementation (no-op)
    void sync() override {}

    // Core processing functions
    void init() override;
    void cleanup() override;

    // Static initialization method
    static void initializeGlobal();



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
    struct FFTPlanPair {
        fftwf_plan forward;
        fftwf_plan backward;
        FFTPlanPair() : forward(nullptr), backward(nullptr) {}
    };
    
   void destroyFFTPlans();

    FFTPlanPair* getPlanPair(const CuboidShape& shape);
    
    std::map<CuboidShape, FFTPlanPair> planMap;
    mutable std::mutex backendMutex;

};



// Concrete CPU Backend Implementation
class CPUBackend : public IBackend {
private:

    // Constructor for external ownership (references to externally-owned components)
    CPUBackend(CPUDeconvolutionBackend& deconv,
               CPUBackendMemoryManager& mem)
        : deconvBackend(deconv),
          memoryManager(mem),
          owner(deconv, mem) {}

    // Constructor for self-ownership (takes ownership of both components)
    CPUBackend(std::unique_ptr<CPUDeconvolutionBackend> deconv,
               std::unique_ptr<CPUBackendMemoryManager> mem)
        : deconvBackend(*deconv),
          memoryManager(*mem),
          owner(std::move(deconv), std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of deconv, external memory)
    CPUBackend(std::unique_ptr<CPUDeconvolutionBackend> deconv,
               CPUBackendMemoryManager& mem)
        : deconvBackend(*deconv),
          memoryManager(mem),
          owner(std::move(deconv), mem) {}

    CPUDeconvolutionBackend& deconvBackend;
    CPUBackendMemoryManager& memoryManager;
    Owner owner;  // Always uses unique_ptr, nullptr for non-owned components



    // Type-safe factory methods for different ownership models
    
    // Create CPUBackend with external ownership (references to externally-owned components)
    static std::shared_ptr<CPUBackend> createWithExternalOwnership(
        CPUDeconvolutionBackend& deconv,
        CPUBackendMemoryManager& mem) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(deconv, mem));
    }

    // Create CPUBackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<CPUBackend> createWithSelfOwnership(
        std::unique_ptr<CPUDeconvolutionBackend> deconv,
        std::unique_ptr<CPUBackendMemoryManager> mem) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(std::move(deconv), std::move(mem)));
    }

    // Create CPUBackend with mixed ownership (takes ownership of deconv, external memory)
    static std::shared_ptr<CPUBackend> createWithMixedOwnership(
        std::unique_ptr<CPUDeconvolutionBackend> deconv,
        CPUBackendMemoryManager& mem) {
        return std::shared_ptr<CPUBackend>(new CPUBackend(std::move(deconv), mem));
    }

public:
    // Factory method to create CPUBackend with individual memory manager
    static CPUBackend* create() {
        auto deconv = std::make_unique<CPUDeconvolutionBackend>();
        auto memoryManager = std::make_unique<CPUBackendMemoryManager>();
        return new CPUBackend(std::move(deconv), std::move(memoryManager));
    }

        
    void sync() override {}
    // Implementation of pure virtual methods
    std::string getDeviceString() const noexcept override {
        return "cpu";
    }
    int getNumberDevices() const noexcept override{
        return 1;
    }

    // Ownership query methods
    bool ownsDeconvolutionBackend() const noexcept override {
        return owner.ownsDeconvBackend();
    }

    bool ownsMemoryManager() const noexcept override {
        return owner.ownsMemoryManager();
    }

    void releaseBackend() override{
        
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
    std::shared_ptr<IBackend> onNewThread(std::shared_ptr<IBackend> original) const override {
        return original;
    }

    std::shared_ptr<IBackend> onNewThreadSharedMemory(std::shared_ptr<IBackend> original) const override {
        return original;
    }

    void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads) const override{
        ioThreads = totalThreads;
        workerThreads = static_cast<size_t>(2*totalThreads/3);
        workerThreads = workerThreads == 0 ? 1 : workerThreads;
    }
};

