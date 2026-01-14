#include "CUDABackend.h"
#include "CUDABackendManager.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cassert>

#include <dolphinbackend/Exceptions.h>


// Unified CUDA error check macro
#define CUDA_CHECK(err, operation) { \
    if (err != cudaSuccess) { \
        throw dolphin::backend::BackendException( \
            std::string("CUDA error: ") + cudaGetErrorString(err), \
            "CUDA", \
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




extern "C" IDeconvolutionBackend* createDeconvolutionBackend() {
    return new CUDADeconvolutionBackend();
}

extern "C" IBackendMemoryManager* createBackendMemoryManager() {
    return new CUDABackendMemoryManager();
}

extern "C" IBackend* createBackend(){
    return CUDABackend::create();
}


// CUDABackendMemoryManager implementation
CUDABackendMemoryManager::CUDABackendMemoryManager(){
}

CUDABackendMemoryManager::~CUDABackendMemoryManager() {
}

void CUDABackendMemoryManager::setMemoryLimit(size_t maxMemorySize) {
    std::unique_lock<std::mutex> lock(device.memory->memoryMutex);
    device.memory->maxMemorySize = maxMemorySize;
}

void CUDABackendMemoryManager::waitForMemory(size_t requiredSize) const {
    std::unique_lock<std::mutex> lock(device.memory->memoryMutex);
    if ((device.memory->totalUsedMemory + requiredSize) > device.memory->maxMemorySize){
        std::cerr << "CUDABackend out of device.memory-> waiting for memory to free up" << std::endl;
    }
    device.memory->memoryCondition.wait(lock, [this, requiredSize]() {
        return device.memory->maxMemorySize == 0 || (device.memory->totalUsedMemory + requiredSize) <= device.memory->maxMemorySize;
    });
}

bool CUDABackendMemoryManager::isOnDevice(void* ptr) const {
    if (ptr == nullptr) return false;
    
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    
    if (result == cudaSuccess) {
        return (attributes.type == cudaMemoryTypeDevice ||
                attributes.type == cudaMemoryTypeManaged);
    } else if (result == cudaErrorInvalidValue) {
        cudaGetLastError(); // Clear the error
        return false;
    } else {
        CUDA_CHECK(result, "isOnDevice");
        return false; // Never reached
    }
}



void CUDABackendMemoryManager::memCopy(const ComplexData& srcData, ComplexData& destData) const {
    // Ensure destination has memory allocated
    if (destData.data == nullptr) {
        allocateMemoryOnDevice(destData);
    }
    
    // Check if sizes match
    BACKEND_CHECK(srcData.size.volume == destData.size.volume, "Size mismatch in memCopy", "CUDA", "memCopy");
    
    // Setup cudaMemcpy3D parameters
    cudaMemcpy3DParms copyParams = {0};
    
    // Source parameters
    copyParams.srcPtr = make_cudaPitchedPtr(
        srcData.data,                           // Source pointer
        srcData.size.width * sizeof(complex),  // Pitch (row width in bytes)
        srcData.size.width,                     // Width in elements
        srcData.size.height                     // Height in elements
    );
    copyParams.srcPos = make_cudaPos(0, 0, 0); // Start from origin
    
    // Destination parameters
    copyParams.dstPtr = make_cudaPitchedPtr(
        destData.data,                          // Destination pointer
        destData.size.width * sizeof(complex), // Pitch (row width in bytes)
        destData.size.width,                    // Width in elements
        destData.size.height                    // Height in elements
    );
    copyParams.dstPos = make_cudaPos(0, 0, 0); // Start from origin
    
    // Copy extent (how much to copy)
    copyParams.extent = make_cudaExtent(
        srcData.size.width * sizeof(complex),  // Width in bytes
        srcData.size.height,                    // Height in elements
        srcData.size.depth                      // Depth in elements
    );
    
    // Determine copy direction
    bool srcIsDevice = isOnDevice(srcData.data);
    bool dstIsDevice = isOnDevice(destData.data);
    
    if (srcIsDevice && dstIsDevice) {
        copyParams.kind = cudaMemcpyDeviceToDevice;
    } else if (!srcIsDevice && dstIsDevice) {
        copyParams.kind = cudaMemcpyHostToDevice;
    } else if (srcIsDevice && !dstIsDevice) {
        copyParams.kind = cudaMemcpyDeviceToHost;
    } else {
        copyParams.kind = cudaMemcpyHostToHost;
    }
    
    // Execute the copy
    cudaError_t err = cudaMemcpy3DAsync(&copyParams, stream);
    CUDA_CHECK(err, "memCopy");
    cudaStreamSynchronize(stream);
    destData.backend = this;
}

ComplexData CUDABackendMemoryManager::allocateMemoryOnDevice(const RectangleShape& shape) const {
    ComplexData result{this, nullptr, shape};
    allocateMemoryOnDevice(result); 
    return result;
}

void CUDABackendMemoryManager::allocateMemoryOnDevice(ComplexData& data) const {
    if (data.data != nullptr && isOnDevice(data.data)) {
        return; // Already on device
    }
    
    // Allocate CUDA memory with unified exception handling
    size_t requested_size = data.size.volume * sizeof(complex);
    
    // Wait for memory if max memory limit is set
    waitForMemory(requested_size);

    void* devicePtr = nullptr;
    cudaError_t err = cudaMallocAsync(&devicePtr, requested_size, stream);
    if (err != cudaSuccess){
        MEMORY_ALLOC_CHECK(data.data, requested_size, "CUDA", "allocateMemoryOnDevice");
    }
    data.data = static_cast<complex*>(devicePtr);
    cudaStreamSynchronize(stream);

    // Update memory tracking
    {
        std::unique_lock<std::mutex> lock(device.memory->memoryMutex);
        device.memory->totalUsedMemory += requested_size;
    }
    
    data.backend = this;
}

ComplexData CUDABackendMemoryManager::copyDataToDevice(const ComplexData& srcdata) const {
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromHostToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth,
                                               destdata.data, srcdata.data, stream);
    }
    destdata.backend = this;

    cudaStreamSynchronize(stream);
    return destdata;
}

ComplexData CUDABackendMemoryManager::moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const {
    if (&destBackend == this){
        return srcdata;
    }
 
    ComplexData destdata = destBackend.allocateMemoryOnDevice(srcdata.size);

    if (srcdata.data != nullptr){
        CUBE_UTL_COPY::copyDataFromDeviceToHost(srcdata.size.width, srcdata.size.height, srcdata.size.depth, destdata.data, srcdata.data, stream);
    
    }
    cudaStreamSynchronize(stream);

    return destdata;
}

ComplexData CUDABackendMemoryManager::copyData(const ComplexData& srcdata) const {
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromDeviceToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth,
                                                 destdata.data, srcdata.data, stream);
    }
    cudaStreamSynchronize(stream);
    return destdata;
}



void CUDABackendMemoryManager::freeMemoryOnDevice(ComplexData& srcdata) const {
    BACKEND_CHECK(srcdata.data != nullptr, "Attempting to free null pointer", "CUDA", "freeMemoryOnDevice");
    size_t requested_size = srcdata.size.volume * sizeof(complex);
    
    cudaError_t err = cudaFreeAsync(srcdata.data, stream);
    CUDA_CHECK(err, "freeMemoryOnDevice");
    cudaStreamSynchronize(stream);

    // Update memory tracking
    {
        std::unique_lock<std::mutex> lock(device.memory->memoryMutex);
        if (device.memory->totalUsedMemory < requested_size) {
            device.memory->totalUsedMemory = static_cast<size_t>(0); // this should never happen
        } else {
            device.memory->totalUsedMemory -= requested_size;
        }
        // Notify waiting threads that device.memory->is now available
        device.memory->memoryCondition.notify_all();
    }
    
    srcdata.data = nullptr;
}

size_t CUDABackendMemoryManager::getAvailableMemory() const {
    // For CUDA backend, return available GPU memory
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    CUDA_CHECK(err, "getAvailableMemory");
    
    return freeMem;
}

size_t CUDABackendMemoryManager::getAllocatedMemory() const {
    std::lock_guard<std::mutex> lock(device.memory->memoryMutex);
    return device.memory->totalUsedMemory;
}







CUDADeconvolutionBackend::CUDADeconvolutionBackend(){

//    cudaSetDevice(0);

}

CUDADeconvolutionBackend::~CUDADeconvolutionBackend() {
    destroyFFTPlans();

}

void CUDADeconvolutionBackend::init(){
 }

void CUDADeconvolutionBackend::initializeGlobal(){
}



void CUDADeconvolutionBackend::cleanup(){
    // Clean up CUDA resources if needed
    destroyFFTPlans();
    std::cout << "[STATUS] CUDA backend postprocessing completed" << std::endl;
}

void CUDADeconvolutionBackend::initializePlan(const RectangleShape& shape){
    


    // Allocate temporary memory for plan creation
    size_t tempSize = sizeof(complex) * shape.volume;
    destroyFFTPlans();
    
    // Create forward FFT plan
    CUFFT_CHECK(cufftCreate(&forward), "initializePlan - forward plan creation");
    CUFFT_CHECK(cufftMakePlan3d(forward, shape.depth, shape.height, shape.width, CUFFT_Z2Z, &tempSize), "initializePlan - forward plan setup");
    CUFFT_CHECK(cufftSetStream(forward, stream), "initializePlan - forward plan stream setup");

    
    // Create FFT plan
    CUFFT_CHECK(cufftCreate(&backward), "initializePlan - backward plan creation");
    CUFFT_CHECK(cufftMakePlan3d(backward, shape.depth, shape.height, shape.width, CUFFT_Z2Z, &tempSize), "initializePlan - backward plan setup");
    CUFFT_CHECK(cufftSetStream(backward, stream), "initializePlan - backward plan stream setup");

    planSize = shape;
    std::cout << "[DEBUG] Successfully created cuFFT plans for shape: " 
              << shape.width << "x" << shape.height << "x" << shape.depth << std::endl;
    cudaStreamSynchronize(stream);
}

void CUDADeconvolutionBackend::destroyFFTPlans(){
    if (forward != 0){
        CUFFT_CHECK(cufftDestroy(forward), "destroyFFTPlans - forward plan");
        forward = 0;
    }
    if (backward != 0){
        CUFFT_CHECK(cufftDestroy(backward), "destroyFFTPlans - forward plan");
        backward = 0;
    }

    planSize = RectangleShape(0,0,0);
}


// FFT Operations
void CUDADeconvolutionBackend::forwardFFT(const ComplexData& in, ComplexData& out) const {
    
    
    if (in.size != planSize){
        const_cast<CUDADeconvolutionBackend*>(this)->initializePlan(in.size);        
    } 
    CUFFT_CHECK(cufftExecZ2Z(forward, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_FORWARD), "forwardFFT");
}

// FFT Operations
void CUDADeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    
    
    if (in.size != planSize){
        const_cast<CUDADeconvolutionBackend*>(this)->initializePlan(in.size);        
    } 
    CUFFT_CHECK(cufftExecZ2Z(backward, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_BACKWARD), "backwardFFT");
}

// Shift Operations
void CUDADeconvolutionBackend::octantFourierShift(ComplexData& data) const {
    cudaError_t err = CUBE_FTT::octantFourierShiftFftwComplex(data.size.width, data.size.height, data.size.depth, data.data, stream);
    CUDA_CHECK(err, "octantFourierShift");
}

void CUDADeconvolutionBackend::inverseQuadrantShift(ComplexData& data) const {
    int width = data.size.width;
    int height = data.size.height;
    int depth = data.size.depth;
    
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < halfHeight; ++y) {
            for (int x = 0; x < halfWidth; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x + halfWidth);

                std::swap(data.data[idx1][0], data.data[idx2][0]);
                std::swap(data.data[idx1][1], data.data[idx2][1]);
            }
        }
    }

    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < halfHeight; ++y) {
            for (int x = halfWidth; x < width; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x - halfWidth);

                std::swap(data.data[idx1][0], data.data[idx2][0]);
                std::swap(data.data[idx1][1], data.data[idx2][1]);
            }
        }
    }

    for (int z = 0; z < halfDepth; ++z) {
        for (int y = halfHeight; y < height; ++y) {
            for (int x = 0; x < halfWidth; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x + halfWidth);

                std::swap(data.data[idx1][0], data.data[idx2][0]);
                std::swap(data.data[idx1][1], data.data[idx2][1]);
            }
        }
    }

    for (int z = 0; z < halfDepth; ++z) {
        for (int y = halfHeight; y < height; ++y) {
            for (int x = halfWidth; x < width; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x - halfWidth);

                std::swap(data.data[idx1][0], data.data[idx2][0]);
                std::swap(data.data[idx1][1], data.data[idx2][1]);
            }
        }
    }
}



// Complex Arithmetic Operations
void CUDADeconvolutionBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexMultiplication", "CUDA", "complexMultiplication");
    cudaError_t err = CUBE_MAT::complexElementwiseMatMulFftwComplex(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, stream);
    CUDA_CHECK(err, "complexMultiplication");
}

void CUDADeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexDivision", "CUDA", "complexDivision");
    cudaError_t err = CUBE_MAT::complexElementwiseMatDivFftwComplex(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, epsilon, stream);
    CUDA_CHECK(err, "complexDivision");
}

void CUDADeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexAddition", "CUDA", "complexAddition");
    for (int i = 0; i < a.size.volume; ++i) {
        result.data[i][0] = a.data[i][0] + b.data[i][0];
        result.data[i][1] = a.data[i][1] + b.data[i][1];
    }
}

void CUDADeconvolutionBackend::scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) const {
    BACKEND_CHECK(a.size.volume == result.size.volume, "Size mismatch in scalarMultiplication", "CUDA", "scalarMultiplication");
    for (int i = 0; i < a.size.volume; ++i) {
        result.data[i][0] = a.data[i][0] * scalar;
        result.data[i][1] = a.data[i][1] * scalar;
    }
}

void CUDADeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexMultiplicationWithConjugate", "CUDA", "complexMultiplicationWithConjugate");
    cudaError_t err = CUBE_MAT::complexElementwiseMatMulConjugateFftwComplex(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, stream);
    CUDA_CHECK(err, "complexMultiplicationWithConjugate");
}

void CUDADeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexDivisionStabilized", "CUDA", "complexDivisionStabilized");
    cudaError_t err = CUBE_MAT::complexElementwiseMatDivStabilizedFftwComplex(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, epsilon, stream);
    CUDA_CHECK(err, "complexDivisionStabilized");
}





// Specialized Functions
void CUDADeconvolutionBackend::hasNAN(const ComplexData& data) const {
    // Implementation would go here
    std::cout << "[DEBUG] hasNAN called on CUDA backend" << std::endl;
}

void CUDADeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
    cudaError_t err = CUBE_REG::calculateLaplacianFftwComplex(psf.size.width, psf.size.height, psf.size.depth, psf.data, laplacian.data, stream);
    CUDA_CHECK(err, "calculateLaplacianOfPSF");
}

void CUDADeconvolutionBackend::normalizeImage(ComplexData& resultImage, double epsilon) const {
    cudaError_t err = CUBE_FTT::normalizeFftwComplexData(1, 1, 1, resultImage.data, stream);
    CUDA_CHECK(err, "normalizeImage");
}

void CUDADeconvolutionBackend::rescaledInverse(ComplexData& data, double cubeVolume) const {
    for (int i = 0; i < data.size.volume; ++i) {
        data.data[i][0] /= cubeVolume;
        data.data[i][1] /= cubeVolume;
    }
}

// Gradient and TV Functions
void CUDADeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    cudaError_t err = CUBE_REG::gradXFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradX.data, stream);
    CUDA_CHECK(err, "gradientX");
}

void CUDADeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    cudaError_t err = CUBE_REG::gradYFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradY.data, stream);
    CUDA_CHECK(err, "gradientY");
}

void CUDADeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    cudaError_t err = CUBE_REG::gradZFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradZ.data, stream);
    CUDA_CHECK(err, "gradientZ");
}

void CUDADeconvolutionBackend::computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
    cudaError_t err = CUBE_REG::computeTVFftwComplex(gx.size.width, gx.size.height, gx.size.depth, lambda, gx.data, gy.data, gz.data, tv.data, stream);
    CUDA_CHECK(err, "computeTV");
}

void CUDADeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) const {
    cudaError_t err = CUBE_REG::normalizeTVFftwComplex(gradX.size.width, gradX.size.height, gradX.size.depth, gradX.data, gradY.data, gradZ.data, epsilon, stream);
    CUDA_CHECK(err, "normalizeTV");
}

// this should be run on the host thread that will use this backend
std::shared_ptr<IBackend> CUDABackend::onNewThread(std::shared_ptr<IBackend> original) const {
    // For CUDA, use the backend manager to get the appropriate backend for this thread
    // This might return the original or a different backend depending on the manager's logic
    std::shared_ptr<CUDABackend> threadBackend = CUDABackendManager::getInstance().getBackendForCurrentThread();
    threadBackend->configureThreadLocalDevice(); // cudasetdevice

    return threadBackend;
}

void CUDABackend::releaseBackend(){
    sync();
    CUDABackendManager::getInstance().releaseBackendForCurrentThread(this);
}