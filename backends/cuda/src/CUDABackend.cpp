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

#include "CUDABackend.h"
#include "CUDABackendManager.h"
// #include <ioconfig.stream>
// #include <sconfig.stream>
#include <cassert>




LogCallback g_logger_cuda;



// CUDABackendMemoryManager implementation
CUDABackendMemoryManager::CUDABackendMemoryManager(CUDABackendConfig config) : config(config) {
}

CUDABackendMemoryManager::~CUDABackendMemoryManager() {
}

void CUDABackendMemoryManager::setMemoryLimit(size_t maxMemorySize) {
    auto access = getMemoryTracking()->getAccess();
    access.data.maxMemorySize = maxMemorySize;
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


// TODO do i want this memCopy or the normal other one
// void CUDABackendMemoryManager::memCopy(const ComplexData& srcData, ComplexData& destData) const {
//     // Validate input data
//     BACKEND_CHECK(srcData.size.getVolume() > 0, "Invalid source data size in memCopy", "CUDA", "memCopy");
//     BACKEND_CHECK(destData.size.getVolume() > 0, "Invalid destination data size in memCopy", "CUDA", "memCopy");
//
//     // Check if sizes match
//     BACKEND_CHECK(srcData.size.getVolume() == destData.size.getVolume(), "Size mismatch in memCopy", "CUDA", "memCopy");
//
//
//     // Setup cudaMemcpy3D parameters
//     cudaMemcpy3DParms copyParams = {0};
//
//     // Source parameters
//     copyParams.srcPtr = make_cudaPitchedPtr(
//         srcData.data,                           // Source pointer
//         srcData.size.width * sizeof(complex_t),  // Pitch (row width in bytes)
//         srcData.size.width,                     // Width in elements
//         srcData.size.height                     // Height in elements
//     );
//     copyParams.srcPos = make_cudaPos(0, 0, 0); // Start from origin
//
//     // Destination parameters
//     copyParams.dstPtr = make_cudaPitchedPtr(
//         destData.data,                          // Destination pointer
//         destData.size.width * sizeof(complex_t), // Pitch (row width in bytes)
//         destData.size.width,                    // Width in elements
//         destData.size.height                    // Height in elements
//     );
//     copyParams.dstPos = make_cudaPos(0, 0, 0); // Start from origin
//
//     // Copy extent (how much to copy)
//     copyParams.extent = make_cudaExtent(
//         srcData.size.width * sizeof(complex_t),  // Width in bytes
//         srcData.size.height,                    // Height in elements
//         srcData.size.depth                      // Depth in elements
//     );
//
//     // Determine copy direction
//     bool srcIsDevice = isOnDevice(srcData.data);
//     bool dstIsDevice = isOnDevice(destData.data);
//
//     if (srcIsDevice && dstIsDevice) {
//         copyParams.kind = cudaMemcpyDeviceToDevice;
//     } else if (!srcIsDevice && dstIsDevice) {
//         copyParams.kind = cudaMemcpyHostToDevice;
//     } else if (srcIsDevice && !dstIsDevice) {
//         copyParams.kind = cudaMemcpyDeviceToHost;
//     } else {
//         copyParams.kind = cudaMemcpyHostToHost;
//     }
//
//     // Execute the copy
//     cudaError_t err = cudaMemcpy3DAsync(&copyParams, config.stream);
//     CUDA_CHECK(err, "memCopy - cudaMemcpy3DAsync");
//
//     err = cudaStreamSynchronize(config.stream);
//     CUDA_CHECK(err, "memCopy - cudaStreamSynchronize");
//
//     destData.backend = this;
// }
//
//
RealData CUDABackendMemoryManager::allocateMemoryOnDeviceReal(const CuboidShape& shape) const{
    RealData result{ this, nullptr, shape };
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}

ComplexData CUDABackendMemoryManager::allocateMemoryOnDevice(const CuboidShape& shape) const{
    CuboidShape complexShape = shape;
    complexShape.width = complexShape.width / 2 + 1;//TODO this is the shape that is needed in the fftw representation of real valued data in complex space
    ComplexData result{ this, nullptr, complexShape };
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}

void* CUDABackendMemoryManager::allocateMemoryOnDevice(size_t requested_size) const {
    // Wait for memory if max memory limit is set
    // waitForMemory(requested_size);

    void* devicePtr = nullptr;
    cudaError_t err = cudaMallocAsync(&devicePtr, requested_size, config.stream);

    CUDA_MEMORY_ALLOC_CHECK(err, requested_size, "allocateMemoryOnDevice - cudaMallocAsync");

    // Synchronize to ensure allocation is complete
    err = cudaStreamSynchronize(config.stream);


    CUDA_CHECK(err, "allocateMemoryOnDevice - cudaStreamSynchronize");

    // Update memory tracking using getAccess()
    auto access = getMemoryTracking()->getAccess();
    access.data.totalUsedMemory += requested_size;

    return devicePtr;
}



void* CUDABackendMemoryManager::copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const {
    void* dest = allocateMemoryOnDevice(size);

    cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, config.stream);
    cudaStreamSynchronize(config.stream);

    return dest;


}

void* CUDABackendMemoryManager::moveDataFromDevice(void* src, size_t size, const CuboidShape& shape, const IBackendMemoryManager& destBackend) const {
    if (&destBackend == this){
        return src;
    }
    void* dest = destBackend.allocateMemoryOnDevice(size);

    cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, config.stream);
    cudaStreamSynchronize(config.stream);

    return dest;
}

void CUDABackendMemoryManager::memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const {
    //TODO make check if all is on device
    cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, config.stream);
    cudaStreamSynchronize(config.stream);
}

void CUDABackendMemoryManager::freeMemoryOnDevice(void* ptr, size_t size) const {
    BACKEND_CHECK(ptr != nullptr, "Attempting to free null pointer", "CUDA", "freeMemoryOnDevice");
    BACKEND_CHECK(size > 0, "Invalid data size for deallocation", "CUDA", "freeMemoryOnDevice");

    cudaError_t err = cudaFreeAsync(ptr, config.stream);
    CUDA_CHECK(err, "freeMemoryOnDevice - cudaFreeAsync");

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "freeMemoryOnDevice - cudaStreamSynchronize");

    // Update memory tracking using getAccess()
    auto access = getMemoryTracking()->getAccess();
    if (access.data.totalUsedMemory < size) {
        access.data.totalUsedMemory = static_cast<size_t>(0); // this should never happen
        g_logger_cuda(std::format("Memory tracking inconsistency detected in freeMemoryOnDevice"), LogLevel::WARN);
    } else {
        access.data.totalUsedMemory -= size;
    }

    ptr = nullptr;
}


size_t CUDABackendMemoryManager::getAvailableMemory() const {
    cudaError_t sync_err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(sync_err, "getAvailableMemory - sync");

    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    CUDA_CHECK(err, "getAvailableMemory - cudaMemGetInfo");

    if (freeMem > totalMem) {
        g_logger_cuda(std::format("Available memory ({}) exceeds total memory ({})", freeMem, totalMem), LogLevel::WARN);
        return 0; // Return 0 to indicate error condition
    }

    return freeMem;
}

size_t CUDABackendMemoryManager::getAllocatedMemory() const {
    auto access = getMemoryTracking()->getAccess();
    return access.data.totalUsedMemory;
}





CUDADeconvolutionBackend::CUDADeconvolutionBackend(CUDABackendConfig config) : config(config){

}

CUDADeconvolutionBackend::~CUDADeconvolutionBackend() {
    destroyPlans();

}


cufftHandle* CUDADeconvolutionBackend::getPlan(const PlanDescription& description) {

    for (cuFFTPlan& plan : cuFFTPlans){
        if (plan.description == description){
            return &plan.plan;
        }
    }


    // Create new plan and store it in the map
    cufftHandle newPlan = initializePlan(description);
    cuFFTPlan plan{ std::move(newPlan), description };
    cuFFTPlans.push_back(std::move(plan));
    return &cuFFTPlans.back().plan;  // Return reference to the stored plan
}

void CUDADeconvolutionBackend::createPlanComplexToReal(cufftHandle& plan, const PlanDescription& description) const {
    size_t tempSize = sizeof(complex_t) * description.shape.depth * description.shape.height * description.shape.width / 2 + 1;
    CUFFT_CHECK(cufftMakePlan3d(plan, description.shape.depth, description.shape.height, description.shape.width, CUFFT_C2R, &tempSize), "getPlan - C2C plan setup");
}

void CUDADeconvolutionBackend::createPlanRealToComplex(cufftHandle& plan, const PlanDescription& description) const {
    size_t tempSize = sizeof(complex_t) * description.shape.depth * description.shape.height * description.shape.width / 2 + 1;
    CUFFT_CHECK(cufftMakePlan3d(plan, description.shape.depth, description.shape.height, description.shape.width, CUFFT_R2C, &tempSize), "getPlan - C2C plan setup");
}

void CUDADeconvolutionBackend::createPlanComplex(cufftHandle& plan, const PlanDescription& description) const {
    size_t tempSize = sizeof(complex_t) * description.shape.getVolume();
    CUFFT_CHECK(cufftMakePlan3d(plan, description.shape.depth, description.shape.height, description.shape.width, CUFFT_C2C, &tempSize), "getPlan - C2C plan setup");
}

cufftHandle CUDADeconvolutionBackend::initializePlan(const PlanDescription& description){
    // Plan not found, create a new one
    CuboidShape shape = description.shape;

    // Validate input shape
    BACKEND_CHECK(shape.getVolume() > 0, "Invalid shape for FFT plan initialization", "CUDA", "getPlan");
    BACKEND_CHECK(shape.width > 0 && shape.height > 0 && shape.depth > 0,
                  "Invalid dimensions for FFT plan initialization", "CUDA", "getPlan");


    cufftHandle newPlan = 0;

    try {
        CUFFT_CHECK(cufftCreate(&newPlan), "getPlan - plan creation");

        if (description.type == PlanType::COMPLEX) createPlanComplex(newPlan, description);
        else if (description.type == PlanType::REAL && description.direction == PlanDirection::FORWARD) createPlanRealToComplex(newPlan, description);
        else if (description.type == PlanType::REAL && description.direction == PlanDirection::BACKWARD) createPlanComplexToReal(newPlan, description);
        assert(newPlan != 0 && "cufft plan not created");

        CUFFT_CHECK(cufftSetStream(newPlan, config.stream), "getPlan - stream setup");

        g_logger_cuda(std::format("Successfully created cuFFT plan for shape: {}x{}x{} direction: {} type: {}",
            shape.width, shape.height, shape.depth,
            (description.direction == PlanDirection::FORWARD ? "FORWARD" : "BACKWARD"),
            (description.type == PlanType::REAL ? "REAL" : "COMPLEX")), LogLevel::DEBUG);

        // Synchronize to ensure plan is ready
        cudaError_t err = cudaStreamSynchronize(config.stream);
        CUDA_CHECK(err, "getPlan - cudaStreamSynchronize");

        return newPlan;
    } catch (...) {
        // Clean up plan if creation fails
        if (newPlan != 0) {
            cufftDestroy(newPlan);
        }
        throw;
    }
}



void CUDADeconvolutionBackend::destroyPlans(){
    for (auto& plan : cuFFTPlans) {
        if (plan.plan != 0) {
            CUFFT_CHECK(cufftDestroy(plan.plan), "destroyFFTPlans - plan destruction");
            plan.plan = 0;
        }
    }
    cuFFTPlans.clear();
}

// FFT Operations
void CUDADeconvolutionBackend::forwardFFT(const ComplexData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.size.getVolume() > 0, "Invalid input data size for forwardFFT", "CUDA", "forwardFFT");
    BACKEND_CHECK(out.size.getVolume() > 0, "Invalid output data size for forwardFFT", "CUDA", "forwardFFT");
    BACKEND_CHECK(in.size.getVolume() == out.size.getVolume(), "Size mismatch in forwardFFT", "CUDA", "forwardFFT");

    // Get or create the forward FFT plan
    PlanDescription desc(PlanDirection::FORWARD, PlanType::COMPLEX, in.size);
    cufftHandle* forwardPlan = const_cast<CUDADeconvolutionBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(forwardPlan != 0, "Forward FFT plan not initialized", "CUDA", "forwardFFT");

    CUFFT_CHECK(cufftExecC2C(*forwardPlan, reinterpret_cast<cufftComplex*>(in.data), reinterpret_cast<cufftComplex*>(out.data), CUFFT_FORWARD), "forwardFFT");
}

// FFT Operations
void CUDADeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.size.getVolume() > 0, "Invalid input data size for backwardFFT", "CUDA", "backwardFFT");
    BACKEND_CHECK(out.size.getVolume() > 0, "Invalid output data size for backwardFFT", "CUDA", "backwardFFT");
    BACKEND_CHECK(in.size.getVolume() == out.size.getVolume(), "Size mismatch in backwardFFT", "CUDA", "backwardFFT");

    // Get or create the backward FFT plan
    PlanDescription desc(PlanDirection::BACKWARD, PlanType::COMPLEX, in.size);
    cufftHandle* backwardPlan = const_cast<CUDADeconvolutionBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(backwardPlan != 0, "Backward FFT plan not initialized", "CUDA", "backwardFFT");

    CUFFT_CHECK(cufftExecC2C(*backwardPlan, reinterpret_cast<cufftComplex*>(in.data), reinterpret_cast<cufftComplex*>(out.data), CUFFT_INVERSE), "backwardFFT");

    complex_t normFactor{1.0f / out.size.getVolume(), 1.0f / out.size.getVolume()};//TESTVALUE
    scalarMultiplication(out, normFactor, out); // Add normalization
}

void CUDADeconvolutionBackend::forwardFFT(const RealData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.size.getVolume() > 0, "Invalid input data size for forwardFFTReal", "CUDA", "forwardFFTReal");
    BACKEND_CHECK(out.size.getVolume() > 0, "Invalid output data size for forwardFFTReal", "CUDA", "forwardFFTReal");

    // Get or create the forward FFT plan
    PlanDescription desc(PlanDirection::FORWARD, PlanType::REAL, in.size);
    cufftHandle* forwardPlan = const_cast<CUDADeconvolutionBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(forwardPlan != 0, "forward FFT plan not initialized", "CUDA", "forwardFFTReal");

    CUFFT_CHECK(cufftExecR2C(*forwardPlan, reinterpret_cast<cufftReal*>(in.data), reinterpret_cast<cufftComplex*>(out.data)), "forwardFFTReal");

}

void CUDADeconvolutionBackend::backwardFFT(const ComplexData& in, RealData& out) const {
    // Validate input data
    BACKEND_CHECK(in.size.getVolume() > 0, "Invalid input data size for backwardFFTReal", "CUDA", "backwardFFTReal");
    BACKEND_CHECK(out.size.getVolume() > 0, "Invalid output data size for backwardFFTReal", "CUDA", "backwardFFTReal");

    // Get or create the backward FFT plan
    PlanDescription desc(PlanDirection::BACKWARD, PlanType::REAL, in.size);
    cufftHandle* backwardPlan = const_cast<CUDADeconvolutionBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(backwardPlan != 0, "Backward FFT plan not initialized", "CUDA", "backwardFFTReal");

    CUFFT_CHECK(cufftExecC2R(*backwardPlan, reinterpret_cast<cufftComplex*>(in.data), reinterpret_cast<cufftReal*>(out.data)), "backwardFFTReal");

    real_t normFactor{1.0f / out.size.getVolume()};//TESTVALUE
    scalarMultiplication(out, normFactor, out); // Add normalization
}

// Shift Operations
void CUDADeconvolutionBackend::octantFourierShift(ComplexData& data) const {
    cudaError_t err = CUBE_FTT::octantFourierShift(data.size.width, data.size.height, data.size.depth, data.data, config.stream);
    CUDA_CHECK(err, "octantFourierShift");
}

void CUDADeconvolutionBackend::octantFourierShift(RealData& data) const {
    cudaError_t err = CUBE_FTT::octantFourierShift(data.size.width, data.size.height, data.size.depth, data.data, config.stream);
    CUDA_CHECK(err, "octantFourierShift");
}



void CUDADeconvolutionBackend::complexAddition(complex_t** dataPointer, ComplexData& sums, int nImages, int imageVolume) const {
    cudaError_t err = CUBE_MAT::complexAddition(dataPointer, sums.data, nImages, imageVolume, config.stream);
}

void CUDADeconvolutionBackend::sumToOne(real_t** data, int nImages, int imageVolume) const {
    cudaError_t err = CUBE_MAT::sumToOne(data, nImages, imageVolume, config.stream);
}

// Complex Arithmetic Operations
void CUDADeconvolutionBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    // Validate input data
    BACKEND_CHECK(a.size.getVolume() > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication");
    BACKEND_CHECK(b.size.getVolume() > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication");
    BACKEND_CHECK(result.size.getVolume() > 0, "Invalid output data size for complexMultiplication", "CUDA", "complexMultiplication");
    BACKEND_CHECK(a.size.getVolume() == b.size.getVolume() && a.size.getVolume() == result.size.getVolume(), "Size mismatch in complexMultiplication", "CUDA", "complexMultiplication");

    cudaError_t err = CUBE_MAT::complexElementwiseMatMul(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, config.stream);
    CUDA_CHECK(err, "complexMultiplication");
}

void CUDADeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    // Validate input data
    BACKEND_CHECK(a.size.getVolume() > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(b.size.getVolume() > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(result.size.getVolume() > 0, "Invalid output data size for complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(a.size.getVolume() == b.size.getVolume() && a.size.getVolume() == result.size.getVolume(), "Size mismatch in complexDivision", "CUDA", "complexDivision");
    BACKEND_CHECK(epsilon >= 0.0, "Invalid epsilon value for complexDivision", "CUDA", "complexDivision");

    cudaError_t err = CUBE_MAT::complexElementwiseMatDiv(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, epsilon, config.stream);
    CUDA_CHECK(err, "complexDivision");
}

void CUDADeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.size.getVolume() == b.size.getVolume() && a.size.getVolume() == result.size.getVolume(), "Size mismatch in complexAddition", "CUDA", "complexAddition");
    cudaError_t err = CUBE_MAT::complexAddition(a.size.width, a.size.height, a.size.depth, a.data, b.data , result.data, config.stream);
    CUDA_CHECK(err, "complexAddition");
}



void CUDADeconvolutionBackend::multiplication(const RealData& a, const RealData& b, RealData& result) const{
    BACKEND_CHECK(a.size.getVolume() == result.size.getVolume(), "Size mismatch in elementwiseDivisionReal", "CUDA", "elementwiseMatMulReal");
    cudaError_t err = CUBE_MAT::elementwiseMatMul(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, config.stream);
    CUDA_CHECK(err, "scalarMultiplicationReal");
}
void CUDADeconvolutionBackend::scalarMultiplication(const RealData& a, real_t scalar, RealData& result) const{
    BACKEND_CHECK(a.size.getVolume() == result.size.getVolume(), "Size mismatch in scalarMultiplicationReal", "CUDA", "scalarMultiplicationReal");
    cudaError_t err = CUBE_MAT::scalarMul(a.size.width, a.size.height, a.size.depth, a.data, scalar , result.data, config.stream);
    CUDA_CHECK(err, "scalarMultiplicationReal");
}
void CUDADeconvolutionBackend::division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const{
    BACKEND_CHECK(a.size.getVolume() == result.size.getVolume(), "Size mismatch in elementwiseDivisionReal", "CUDA", "elementwiseDivisionReal");
    cudaError_t err = CUBE_MAT::elementwiseMatDiv(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, epsilon, config.stream);
    CUDA_CHECK(err, "elementwiseDivisionReal");
}

void CUDADeconvolutionBackend::scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
    BACKEND_CHECK(a.size.getVolume() == result.size.getVolume(), "Size mismatch in scalarMultiplication", "CUDA", "scalarMultiplication");
    cudaError_t err = CUBE_MAT::complexScalarMul(a.size.width, a.size.height, a.size.depth, a.data, scalar , result.data, config.stream);
    CUDA_CHECK(err, "scalarMultiplication");
}

void CUDADeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.size.getVolume() == b.size.getVolume() && a.size.getVolume() == result.size.getVolume(), "Size mismatch in complexMultiplicationWithConjugate", "CUDA", "complexMultiplicationWithConjugate");
    cudaError_t err = CUBE_MAT::complexElementwiseMatMulConjugate(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, config.stream);
    CUDA_CHECK(err, "complexMultiplicationWithConjugate");
}

void CUDADeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.size.getVolume() == b.size.getVolume() && a.size.getVolume() == result.size.getVolume(), "Size mismatch in complexDivisionStabilized", "CUDA", "complexDivisionStabilized");
    cudaError_t err = CUBE_MAT::complexElementwiseMatDivStabilized(a.size.width, a.size.height, a.size.depth, a.data, b.data, result.data, epsilon, config.stream);
    CUDA_CHECK(err, "complexDivisionStabilized");
}




// Specialized Functions
void CUDADeconvolutionBackend::hasNAN(const ComplexData& data) const {
    // Implementation would go here
    g_logger_cuda(std::format("hasNAN called on CUDA backend"), LogLevel::DEBUG);
}

void CUDADeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
    cudaError_t err = CUBE_REG::calculateLaplacian(psf.size.width, psf.size.height, psf.size.depth, psf.data, laplacian.data, config.stream);
    CUDA_CHECK(err, "calculateLaplacianOfPSF");
}

// void CUDADeconvolutionBackend::normalizeImage(ComplexData& resultImage, real_t epsilon) const {
//     cudaError_t err = CUBE_FTT::normalizeData(1, 1, 1, resultImage.data, config.stream);
//     CUDA_CHECK(err, "normalizeImage");
// }

// void CUDADeconvolutionBackend::rescaledInverse(ComplexData& data, real_t cubeVolume) const {
//     for (int i = 0; i < data.size.getVolume(); ++i) {
//         data.data[i][0] /= cubeVolume;
//         data.data[i][1] /= cubeVolume;
//     }
// }

// Gradient and TV Functions
void CUDADeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    cudaError_t err1 = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err1, "found it");

    cudaError_t err = CUBE_REG::gradX(image.size.width, image.size.height, image.size.depth, image.data, gradX.data, config.stream);
    CUDA_CHECK(err, "gradientX");
}

void CUDADeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    cudaError_t err = CUBE_REG::gradY(image.size.width, image.size.height, image.size.depth, image.data, gradY.data, config.stream);
    CUDA_CHECK(err, "gradientY");
}

void CUDADeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    cudaError_t err = CUBE_REG::gradZ(image.size.width, image.size.height, image.size.depth, image.data, gradZ.data, config.stream);
    CUDA_CHECK(err, "gradientZ");
}

void CUDADeconvolutionBackend::computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
    cudaError_t err = CUBE_REG::computeTV(gx.size.width, gx.size.height, gx.size.depth, lambda, gx.data, gy.data, gz.data, tv.data, config.stream);
    CUDA_CHECK(err, "computeTV");
}

void CUDADeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const {
    cudaError_t err = CUBE_REG::normalizeTV(gradX.size.width, gradX.size.height, gradX.size.depth, gradX.data, gradY.data, gradZ.data, epsilon, config.stream);
    CUDA_CHECK(err, "normalizeTV");
}

