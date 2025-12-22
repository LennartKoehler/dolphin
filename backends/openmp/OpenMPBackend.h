#pragma once
#include "backend/IBackend.h"
#include "backend/IDeconvolutionBackend.h"
#include "backend/IBackendMemoryManager.h"
#include <fftw3.h>
#include <map>
#include <omp.h>



class OpenMPBackendMemoryManager : public IBackendMemoryManager{
public:
    OpenMPBackendMemoryManager();
    
    ~OpenMPBackendMemoryManager();
    
    // Override device type method
    std::string getDeviceType() const noexcept override {
        return "openmp";
    }
    
    // Synchronization method - OpenMP implementation (no-op)
    void sync() override {}
    
    // Memory management initialization
    void setMemoryLimit(size_t maxMemorySize = 0) override;
    
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) const override;
    void allocateMemoryOnDevice(ComplexData& data) const override;
    ComplexData allocateMemoryOnDevice(const RectangleShape& shape) const override;
    bool isOnDevice(void* data) const override;
    ComplexData copyData(const ComplexData& srcdata) const override;
    ComplexData copyDataToDevice(const ComplexData& srcdata) const override; // for openmp these are copy operations
    ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const override; // for openmp these are copy operations
    void freeMemoryOnDevice(ComplexData& data) const override;
    size_t getAvailableMemory() const override;
    size_t getAllocatedMemory() const override;


private:
    // Memory management
    static MemoryTracking memory;
    
    // Helper method to wait for memory availability
    void waitForMemory(size_t requiredSize) const;
          // Static method to get memory tracking instance
    static MemoryTracking& getMemoryTracking() { return memory; }

};

class OpenMPDeconvolutionBackend : public IDeconvolutionBackend{
public:
    OpenMPDeconvolutionBackend();
    ~OpenMPDeconvolutionBackend() override;
    
    // Override device type method
    std::string getDeviceType() const noexcept override {
        return "openmp";
    }

    // Synchronization method - OpenMP implementation (no-op)
    void sync() override {}

    // Core processing functions
    void init() override;
    void cleanup() override;

    // Static initialization method
    static void initializeGlobal();



    void initializePlan(const RectangleShape& cube) override;
     // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) const override;
    void backwardFFT(const ComplexData& in, ComplexData& out) const override;

    // Shift functions
    void octantFourierShift(ComplexData& data) const override;
    void inverseQuadrantShift(ComplexData& data) const override;

    // Complex arithmetic functions
    void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const override;
    void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) const override;
    void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const override;


    void complexMultiplicationAVX2(const ComplexData& a, const ComplexData& b, ComplexData& result) const;
    // Specialized functions
    void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const override;
    void normalizeImage(ComplexData& resultImage, double epsilon) const override;
    void rescaledInverse(ComplexData& data, double cubeVolume) const override;

    // Debug functions
    void hasNAN(const ComplexData& data) const override;

    // Layer and visualization functions
    void reorderLayers(ComplexData& data) const override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) const override;

private:
    struct FFTPlanPair {
        fftw_plan forward;
        fftw_plan backward;
        FFTPlanPair() : forward(nullptr), backward(nullptr) {}
    };
    
   void destroyFFTPlans();

    FFTPlanPair* getPlanPair(const RectangleShape& shape);
    
    std::map<RectangleShape, FFTPlanPair> planMap;
    mutable std::mutex backendMutex;

};



// Concrete OpenMP Backend Implementation
class OpenMPBackend : public IBackend {
private:
    // Constructor for external ownership (references to externally-owned components)
    OpenMPBackend(OpenMPDeconvolutionBackend& deconv,
               OpenMPBackendMemoryManager& mem)
        : deconvBackend(deconv),
          memoryManager(mem),
          owner(deconv, mem) {}

    // Constructor for self-ownership (takes ownership of both components)
    OpenMPBackend(std::unique_ptr<OpenMPDeconvolutionBackend> deconv,
               std::unique_ptr<OpenMPBackendMemoryManager> mem)
        : deconvBackend(*deconv),
          memoryManager(*mem),
          owner(std::move(deconv), std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of deconv, external memory)
    OpenMPBackend(std::unique_ptr<OpenMPDeconvolutionBackend> deconv,
               OpenMPBackendMemoryManager& mem)
        : deconvBackend(*deconv),
          memoryManager(mem),
          owner(std::move(deconv), mem) {}

    OpenMPDeconvolutionBackend& deconvBackend;
    OpenMPBackendMemoryManager& memoryManager;
    Owner owner;  // Always uses unique_ptr, nullptr for non-owned components



    // Type-safe factory methods for different ownership models
    
    // Create OpenMPBackend with external ownership (references to externally-owned components)
    static std::shared_ptr<OpenMPBackend> createWithExternalOwnership(
        OpenMPDeconvolutionBackend& deconv,
        OpenMPBackendMemoryManager& mem) {
        return std::shared_ptr<OpenMPBackend>(new OpenMPBackend(deconv, mem));
    }

    // Create OpenMPBackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<OpenMPBackend> createWithSelfOwnership(
        std::unique_ptr<OpenMPDeconvolutionBackend> deconv,
        std::unique_ptr<OpenMPBackendMemoryManager> mem) {
        return std::shared_ptr<OpenMPBackend>(new OpenMPBackend(std::move(deconv), std::move(mem)));
    }

    // Create OpenMPBackend with mixed ownership (takes ownership of deconv, external memory)
    static std::shared_ptr<OpenMPBackend> createWithMixedOwnership(
        std::unique_ptr<OpenMPDeconvolutionBackend> deconv,
        OpenMPBackendMemoryManager& mem) {
        return std::shared_ptr<OpenMPBackend>(new OpenMPBackend(std::move(deconv), mem));
    }

public:
    // Factory method to create OpenMPBackend with individual memory manager
    static OpenMPBackend* create() {
        auto deconv = std::make_unique<OpenMPDeconvolutionBackend>();
        auto memoryManager = std::make_unique<OpenMPBackendMemoryManager>();
        return new OpenMPBackend(std::move(deconv), std::move(memoryManager));
    }
        
    void sync() override {}
    // Implementation of pure virtual methods
    std::string getDeviceType() const noexcept override {
        return "openmp";
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
            throw std::runtime_error("Cannot release deconvolution backend: not owned by this OpenMPBackend");
        }
        return owner.releaseDeconvBackend();
    }

    std::unique_ptr<IBackendMemoryManager> releaseMemoryManager() override {
        if (!ownsMemoryManager()) {
            throw std::runtime_error("Cannot release memory manager: not owned by this OpenMPBackend");
        }
        return owner.releaseMemoryManager();
    }

    // Take ownership of components
    void takeOwnership(std::unique_ptr<IDeconvolutionBackend> deconv) override {
        if (&(*deconv) != &deconvBackend) {
            throw std::runtime_error("Cannot take ownership: provided deconv backend is not the one currently referenced");
        }
        if (!owner.ownsDeconvBackend()) {
            throw std::runtime_error("Cannot take ownership: deconv backend is not owned by this OpenMPBackend");
        }
        owner.takeOwnership(std::move(deconv));
    }

    void takeOwnership(std::unique_ptr<IBackendMemoryManager> mem) override {
        if (&(*mem) != &memoryManager) {
            throw std::runtime_error("Cannot take ownership: provided memory manager is not the one currently referenced");
        }
        if (!owner.ownsMemoryManager()) {
            throw std::runtime_error("Cannot take ownership: memory manager is not owned by this OpenMPBackend");
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


    
    // Overloaded version for OpenMP: simply return the original since OpenMP doesn't need complex thread management
    std::shared_ptr<IBackend> onNewThread(std::shared_ptr<IBackend> original) const override {
        return original;
    }
};