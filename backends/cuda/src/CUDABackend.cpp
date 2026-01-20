#include "CUDABackend.h"
#include "CUDABackendManager.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cassert>







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
    // Validate input data
    BACKEND_CHECK(srcData.size.volume > 0, "Invalid source data size in memCopy", "CUDA", "memCopy");
    BACKEND_CHECK(destData.size.volume > 0, "Invalid destination data size in memCopy", "CUDA", "memCopy");
    
    // Check if sizes match
    BACKEND_CHECK(srcData.size.volume == destData.size.volume, "Size mismatch in memCopy", "CUDA", "memCopy");
    
    // Ensure destination has memory allocated
    if (destData.data == nullptr) {
        allocateMemoryOnDevice(destData);
    }
    
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
    CUDA_CHECK(err, "memCopy - cudaMemcpy3DAsync");
    
    err = cudaStreamSynchronize(stream);
    CUDA_CHECK(err, "memCopy - cudaStreamSynchronize");
    
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
    
    // Validate data size
    BACKEND_CHECK(data.size.volume > 0, "Invalid data size for allocation", "CUDA", "allocateMemoryOnDevice");
    
    // Allocate CUDA memory with unified exception handling
    size_t requested_size = data.size.volume * sizeof(complex);
    
    // Check for potential overflow
    BACKEND_CHECK(requested_size / sizeof(complex) == data.size.volume,
                  "Size overflow detected in memory allocation", "CUDA", "allocateMemoryOnDevice");
    
    // Wait for memory if max memory limit is set
    waitForMemory(requested_size);

    void* devicePtr = nullptr;
    cudaError_t err = cudaMallocAsync(&devicePtr, requested_size, stream);
    CUDA_CHECK(err, "allocateMemoryOnDevice - cudaMallocAsync");
    
    data.data = static_cast<complex*>(devicePtr);
    
    // Synchronize to ensure allocation is complete
    err = cudaStreamSynchronize(stream);
    CUDA_CHECK(err, "allocateMemoryOnDevice - cudaStreamSynchronize");

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
    
    // Validate data size before freeing
    BACKEND_CHECK(srcdata.size.volume > 0, "Invalid data size for deallocation", "CUDA", "freeMemoryOnDevice");
    
    size_t requested_size = srcdata.size.volume * sizeof(complex);
    
    // Check for potential overflow
    BACKEND_CHECK(requested_size / sizeof(complex) == srcdata.size.volume,
                  "Size overflow detected in memory deallocation", "CUDA", "freeMemoryOnDevice");
    
    cudaError_t err = cudaFreeAsync(srcdata.data, stream);
    CUDA_CHECK(err, "freeMemoryOnDevice - cudaFreeAsync");
    
    err = cudaStreamSynchronize(stream);
    CUDA_CHECK(err, "freeMemoryOnDevice - cudaStreamSynchronize");

    // Update memory tracking
    {
        std::unique_lock<std::mutex> lock(device.memory->memoryMutex);
        if (device.memory->totalUsedMemory < requested_size) {
            device.memory->totalUsedMemory = static_cast<size_t>(0); // this should never happen
            std::cerr << "[WARNING] Memory tracking inconsistency detected in freeMemoryOnDevice" << std::endl;
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
    CUDA_CHECK(err, "getAvailableMemory - cudaMemGetInfo");
    
    if (freeMem > totalMem) {
        std::cerr << "[WARNING] Available memory (" << freeMem
                  << ") exceeds total memory (" << totalMem << ")" << std::endl;
        return 0; // Return 0 to indicate error condition
    }
    
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
    // Validate input shape
    BACKEND_CHECK(shape.volume > 0, "Invalid shape for FFT plan initialization", "CUDA", "initializePlan");
    BACKEND_CHECK(shape.width > 0 && shape.height > 0 && shape.depth > 0,
                  "Invalid dimensions for FFT plan initialization", "CUDA", "initializePlan");
    
    // Check for potential overflow
    size_t tempSize = sizeof(complex) * shape.volume;
    BACKEND_CHECK(tempSize / sizeof(complex) == shape.volume,
                  "Size overflow detected in FFT plan initialization", "CUDA", "initializePlan");
    
    // Destroy existing plans first
    destroyFFTPlans();
    
    try {
        // Create forward FFT plan
        CUFFT_CHECK(cufftCreate(&forward), "initializePlan - forward plan creation");
        CUFFT_CHECK(cufftMakePlan3d(forward, shape.depth, shape.height, shape.width, CUFFT_Z2Z, &tempSize), "initializePlan - forward plan setup");
        CUFFT_CHECK(cufftSetStream(forward, stream), "initializePlan - forward plan stream setup");

        // Create backward FFT plan
        CUFFT_CHECK(cufftCreate(&backward), "initializePlan - backward plan creation");
        CUFFT_CHECK(cufftMakePlan3d(backward, shape.depth, shape.height, shape.width, CUFFT_Z2Z, &tempSize), "initializePlan - backward plan setup");
        CUFFT_CHECK(cufftSetStream(backward, stream), "initializePlan - backward plan stream setup");

        planSize = shape;
        std::cout << "[DEBUG] Successfully created cuFFT plans for shape: "
                  << shape.width << "x" << shape.height << "x" << shape.depth << std::endl;
        
        // Synchronize to ensure plans are ready
        cudaError_t err = cudaStreamSynchronize(stream);
        CUDA_CHECK(err, "initializePlan - cudaStreamSynchronize");
    } catch (...) {
        // Clean up plans if creation fails
        destroyFFTPlans();
        throw;
    }
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
    // Validate input data
    BACKEND_CHECK(in.size.volume > 0, "Invalid input data size for forwardFFT", "CUDA", "forwardFFT");
    BACKEND_CHECK(out.size.volume > 0, "Invalid output data size for forwardFFT", "CUDA", "forwardFFT");
    BACKEND_CHECK(in.size.volume == out.size.volume, "Size mismatch in forwardFFT", "CUDA", "forwardFFT");
    
    if (in.size != planSize){
        const_cast<CUDADeconvolutionBackend*>(this)->initializePlan(in.size);
    }
    
    // Validate FFT plans
    BACKEND_CHECK(forward != 0, "Forward FFT plan not initialized", "CUDA", "forwardFFT");
    
    CUFFT_CHECK(cufftExecZ2Z(forward, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_FORWARD), "forwardFFT");
}

// FFT Operations
void CUDADeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.size.volume > 0, "Invalid input data size for backwardFFT", "CUDA", "backwardFFT");
    BACKEND_CHECK(out.size.volume > 0, "Invalid output data size for backwardFFT", "CUDA", "backwardFFT");
    BACKEND_CHECK(in.size.volume == out.size.volume, "Size mismatch in backwardFFT", "CUDA", "backwardFFT");
    
    if (in.size != planSize){
        const_cast<CUDADeconvolutionBackend*>(this)->initializePlan(in.size);
    }
    
    // Validate FFT plans
    BACKEND_CHECK(backward != 0, "Backward FFT plan not initialized", "CUDA", "backwardFFT");
    
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
    // Validate input data
    BACKEND_CHECK(a.size.volume > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication");
    BACKEND_CHECK(b.size.volume > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication");
    BACKEND_CHECK(result.size.volume > 0, "Invalid output data size for complexMultiplication", "CUDA", "complexMultiplication");
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexMultiplication", "CUDA", "complexMultiplication");
    
    cudaError_t err = CUBE_MAT::complexElementwiseMatMulFftwComplex(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, stream);
    CUDA_CHECK(err, "complexMultiplication");
}

void CUDADeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
    // Validate input data
    BACKEND_CHECK(a.size.volume > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(b.size.volume > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(result.size.volume > 0, "Invalid output data size for complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(a.size.volume == b.size.volume && a.size.volume == result.size.volume, "Size mismatch in complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(epsilon >= 0.0, "Invalid epsilon value for complexDivision", "CUDA", "complexDivision");
    
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

int CUDABackend::getNumberDevices() const {
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    return nDevices; 
}

std::shared_ptr<IBackend> CUDABackend::onNewThreadSharedMemory(std::shared_ptr<IBackend> original) const {
    std::shared_ptr<CUDABackend> threadBackend = CUDABackendManager::getInstance().getBackendForCurrentThreadSameDevice(device);
    threadBackend->configureThreadLocalDevice();
    return threadBackend;
}

void CUDABackend::releaseBackend(){
    sync();
    CUDABackendManager::getInstance().releaseBackendForCurrentThread(this);
}