#pragma once
#include "backend/IBackend.h"
#include "backend/IDeconvolutionBackend.h"
#include "backend/IBackendMemoryManager.h"
#include <cufftw.h>
#include <CUBE.h>
#include <cuda_runtime.h>
#include <map>





class CUDABackendMemoryManager : public IBackendMemoryManager{
public:
    // Constructor
    CUDABackendMemoryManager();
    ~CUDABackendMemoryManager();
    
    void setStream(cudaStream_t stream){ this->stream = stream;}
    // Override device type method
    std::string getDeviceType() const noexcept override {
        return "cuda";
    }
    
    void sync() override {cudaStreamSynchronize(stream);}
    // Memory management initialization
    void setMemoryLimit(size_t maxMemorySize = 0) override;
    
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) const override;
    void allocateMemoryOnDevice(ComplexData& data) const override;
    ComplexData allocateMemoryOnDevice(const RectangleShape& shape) const override;
    bool isOnDevice(void* data) const override;
    ComplexData copyData(const ComplexData& srcdata) const override;
    ComplexData copyDataToDevice(const ComplexData& srcdata) const override; // for gpu these are copy operations
    ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const override; // for gpu these are copy operations
    void freeMemoryOnDevice(ComplexData& data) const override;
    size_t getAvailableMemory() const override;
private:

    static MemoryTracking memory; 
    // CUDA stream for memory operations
    cudaStream_t stream = cudaStreamLegacy;
    
    // Helper method to wait for memory availability
    void waitForMemory(size_t requiredSize) const;
    
    // Static method to get memory tracking instance
    static MemoryTracking& getMemoryTracking() { return memory; }
};

class CUDADeconvolutionBackend : public IDeconvolutionBackend{
public:
    CUDADeconvolutionBackend();
    ~CUDADeconvolutionBackend() override;
    
    // Override device type method
    std::string getDeviceType() const noexcept override {
        return "cuda";
    }

    // Core processing functions
    virtual void init() override;
    virtual void cleanup() override;


    void sync() override {cudaStreamSynchronize(stream);}
    void setStream(cudaStream_t stream){ this->stream = stream;}
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

    // Specialized functions
    void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const override;
    void normalizeImage(ComplexData& resultImage, double epsilon) const override;
    void rescaledInverse(ComplexData& data, double cubeVolume) const override;
    // void saveInterimImages(const ComplexData& resultImage, int gridNum, int channel_z, int i) const override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) const override;

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
    
    
    void destroyFFTPlans();
   
    cufftHandle forward = CUFFT_PLAN_NULL;
    cufftHandle backward = CUFFT_PLAN_NULL;
    RectangleShape planSize;
    cudaStream_t stream = cudaStreamLegacy;
    bool initialized = false;

};


class CUDABackendManager;

// Concrete CUDA Backend Implementation
class CUDABackend : public IBackend {
    friend CUDABackendManager;
private:
    // Constructor for external ownership (references to externally-owned components)
    CUDABackend(CUDADeconvolutionBackend& deconv,
                CUDABackendMemoryManager& mem)
        : deconvBackend(deconv),
          memoryBackend(mem),
          owner(deconv, mem) {}

    // Constructor for self-ownership (takes ownership of both components)
    CUDABackend(std::unique_ptr<CUDADeconvolutionBackend> deconv,
                std::unique_ptr<CUDABackendMemoryManager> mem)
        : deconvBackend(*deconv),
          memoryBackend(*mem),
          owner(std::move(deconv), std::move(mem)) {}

    // Constructor for mixed ownership (takes ownership of deconv, external memory)
    CUDABackend(std::unique_ptr<CUDADeconvolutionBackend> deconv,
                CUDABackendMemoryManager& mem)
        : deconvBackend(*deconv),
          memoryBackend(mem),
          owner(std::move(deconv), mem) {}

    CUDADeconvolutionBackend& deconvBackend;
    CUDABackendMemoryManager& memoryBackend;
    Owner owner;  // Specialized CUDA owner

    // Type-safe factory methods for different ownership models
    
    // Create CUDABackend with external ownership (references to externally-owned components)
    static std::shared_ptr<CUDABackend> createWithExternalOwnership(
        CUDADeconvolutionBackend& deconv,
        CUDABackendMemoryManager& mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(deconv, mem));
    }

    // Create CUDABackend with self-ownership (takes ownership of both components)
    static std::shared_ptr<CUDABackend> createWithSelfOwnership(
        std::unique_ptr<CUDADeconvolutionBackend> deconv,
        std::unique_ptr<CUDABackendMemoryManager> mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(std::move(deconv), std::move(mem)));
    }

    // Create CUDABackend with mixed ownership (takes ownership of deconv, external memory)
    static std::shared_ptr<CUDABackend> createWithMixedOwnership(
        std::unique_ptr<CUDADeconvolutionBackend> deconv,
        CUDABackendMemoryManager& mem) {
        return std::shared_ptr<CUDABackend>(new CUDABackend(std::move(deconv), mem));
    }

public:
    // Factory method to create CUDABackend with
    static CUDABackend* create() {
        auto deconv = std::make_unique<CUDADeconvolutionBackend>();
        auto memoryManager = std::make_unique<CUDABackendMemoryManager>();

        return new CUDABackend(std::move(deconv), std::move(memoryManager));
    }

    // Implementation of pure virtual methods
    std::string getDeviceType() const noexcept override {
        return "cuda";
    }
    void sync() override{
        deconvBackend.sync();
        memoryBackend.sync(); 
    }

    void setStream(cudaStream_t stream){
        deconvBackend.setStream(stream);
        memoryBackend.setStream(stream);
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

    // Clone method - creates a new thread-specific backend
    // The ownership model of the clone depends on the onNewThread() implementation
    std::shared_ptr<IBackend> onNewThread() const override;
    void releaseBackend() override;
};