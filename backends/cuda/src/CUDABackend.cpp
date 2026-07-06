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

#include "cuda_backend/CUDABackend.h"
#include "cuda_backend/CUDABackendManager.h"
// #include <ioconfig.stream>
// #include <sconfig.stream>
#include <cassert>
#include <iostream>
#include <spdlog/fmt/fmt.h>




LogCallback g_logger_cuda =[](const std::string& context, const std::string& message, LogLevel level){
    std::cout << context << ": " << message << std::endl;
};

void CUDABackendMemoryManager::logWithContext(const std::string& msg, LogLevel level) const {
    g_logger_cuda(buildCudaContext(config), msg, level);
}

void CUDAComputeBackend::logWithContext(const std::string& msg, LogLevel level) const {
    g_logger_cuda(buildCudaContext(config), msg, level);
}


// CUDABackendMemoryManager implementation
CUDABackendMemoryManager::CUDABackendMemoryManager(CUDABackendConfig config) : config(config) {
}

CUDABackendMemoryManager::~CUDABackendMemoryManager() {
}

void CUDABackendMemoryManager::setMemoryLimit(size_t maxMemorySize) {
    auto access = getMemoryTracking()->getAccess();
    access.data.maxMemorySize = maxMemorySize;
}


bool CUDABackendMemoryManager::isOnDevice(const void* ptr) const {
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
        CUDA_CHECK(result, "isOnDevice", buildCudaContext(config));
        return false; // Never reached
    }
}


// TODO do i want this memCopy or the normal other one
// void CUDABackendMemoryManager::memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const{
//
//
//     // Setup cudaMemcpy3D parameters
//     cudaMemcpy3DParms copyParams = {0};
//
//     // Source parameters
//     copyParams.srcPtr = make_cudaPitchedPtr(
//         srcData.getData(),                           // Source pointer
//         srcData.getSize().width * sizeof(complex_t),  // Pitch (row width in bytes)
//         srcData.getSize().width,                     // Width in elements
//         srcData.getSize().height                     // Height in elements
//     );
//     copyParams.srcPos = make_cudaPos(0, 0, 0); // Start from origin
//
//     // Destination parameters
//     copyParams.dstPtr = make_cudaPitchedPtr(
//         destData.getData(),                          // Destination pointer
//         destData.getSize().width * sizeof(complex_t), // Pitch (row width in bytes)
//         destData.getSize().width,                    // Width in elements
//         destData.getSize().height                    // Height in elements
//     );
//     copyParams.dstPos = make_cudaPos(0, 0, 0); // Start from origin
//
//     // Copy extent (how much to copy)
//     copyParams.extent = make_cudaExtent(
//         srcData.getSize().width * sizeof(complex_t),  // Width in bytes
//         srcData.getSize().height,                    // Height in elements
//         srcData.getSize().depth                      // Depth in elements
//     );
//
//     // Determine copy direction
//     bool srcIsDevice = isOnDevice(srcData.getData());
//     bool dstIsDevice = isOnDevice(destData.getData());
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
//     CUDA_CHECK(err, "memCopy - cudaMemcpy3DAsync", buildCudaContext(config));
//
//     err = cudaStreamSynchronize(config.stream);
//     CUDA_CHECK(err, "memCopy - cudaStreamSynchronize", buildCudaContext(config));
//
//     destData.backend = this;
// }

RealData CUDABackendMemoryManager::allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const{
    logWithContext(fmt::format("allocateMemoryOnDeviceRealFFTInPlace for shape: {}", shape.print()), LogLevel::DEBUG);
    CuboidShape shapeForInplaceFFT = shape;
    shapeForInplaceFFT.width = 2 *(shapeForInplaceFFT.width/2 + 1);
    size_t padding = shapeForInplaceFFT.width - shape.width;
    std::size_t bytes = shapeForInplaceFFT.getVolume() * sizeof(real_t);

    // memory should be larger than shape, as there is some padding at the end
    RealData result{ this, nullptr, shape, shape, bytes, padding};

    // padded stride for FFTW in-place r2c set via constructor
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}
RealData CUDABackendMemoryManager::allocateMemoryOnDeviceReal(const CuboidShape& shape) const {
    logWithContext(fmt::format("allocateMemoryOnDeviceReal for shape: {}", shape.print()), LogLevel::DEBUG);
    std::size_t bytes = shape.getVolume() * sizeof(real_t);

    RealData result{ this, nullptr, shape, shape, bytes, 0};
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}


ComplexData CUDABackendMemoryManager::allocateMemoryOnDeviceComplex(const CuboidShape& shape) const{
    logWithContext(fmt::format("allocateMemoryOnDeviceComplex for shape: {}", shape.print()), LogLevel::DEBUG);
    CuboidShape complexShape = shape;
    complexShape.width = complexShape.width / 2 + 1;//TODO this is the shape that is needed in the fftw representation of real valued data in complex space
    CuboidShape originalShape = shape;
    ComplexData result{ this, nullptr, complexShape, originalShape, complexShape.getVolume() * sizeof(complex_t), 0};

    // No extra padding: stride equals the complex width
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}
ComplexData CUDABackendMemoryManager::allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const{
    logWithContext(fmt::format("allocateMemoryOnDeviceComplexFull for shape: {}", shape.print()), LogLevel::DEBUG);
    ComplexData result{ this, nullptr, shape, shape, shape.getVolume() * sizeof(complex_t), 0};
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}
DataView<real_t> CUDABackendMemoryManager::reinterpret(ComplexData& data) const{
    CuboidShape realShape = data.getRealSize();
    // The padding field on ComplexData stores the real_t padding value P_real = 2*(W/2+1) - W,
    // which allows recovering the original real width: real_width = 2*complex_width - P_real = W.
    // The same P_real value is the correct real_t padding for the resulting RealView.
    CuboidShape shapeForInplaceFFT = realShape;
    shapeForInplaceFFT.width = 2 *(shapeForInplaceFFT.width/2 + 1);
    size_t padding = shapeForInplaceFFT.width - realShape.width;

    DataView<real_t> result = DataView<real_t>{data.getBackend(), reinterpret_cast<real_t*>(data.getData()), realShape, realShape, data.getDataBytes(), padding};
    data.setBackend(nullptr); // so it doesnt delete the data
    return result;
}

DataView<complex_t> CUDABackendMemoryManager::reinterpret(RealData& data) const{
    CuboidShape complexShape = data.getSize();
    complexShape.width = complexShape.width / 2 + 1;//TODO this is the shape that is needed in the fftw representation of real valued data in complex space

    // Keep the real padding value as-is in the complex view's padding field.
    // Although this is technically in real_t units (not complex_t), it serves as metadata
    // that allows reinterpret(ComplexData -> RealView) to correctly recover the original
    // real width via: real_width = 2 * complex_width - padding = W.
    // The ComplexView's convertIndex() is never used in the FFT path (raw pointers are
    // used instead), so the incorrect padding units don't cause issues in practice.
    DataView<complex_t> result = DataView<complex_t>{data.getBackend(), reinterpret_cast<complex_t*>(data.getData()), complexShape, data.getSize(), data.getDataBytes(), 0};
    data.setBackend(nullptr); // so it doesnt delete the data
    return result;
}



// RealData CUDABackendMemoryManager::allocateMemoryOnDeviceReal(const CuboidShape& shape) const{
//     RealData result{ this, nullptr, shape, shape, shape.getVolume() * sizeof(real_t), 0};
//     IBackendMemoryManager::allocateMemoryOnDevice(result);
//     return result;
// }
//
// RealData CUDABackendMemoryManager::allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const{
//     RealData result{ this, nullptr, shape, shape, shape.getVolume() * sizeof(real_t), 0};
//     IBackendMemoryManager::allocateMemoryOnDevice(result);
//     return result;
// }
//
// ComplexData CUDABackendMemoryManager::allocateMemoryOnDeviceComplex(const CuboidShape& shape) const{
//     CuboidShape complexShape = shape;
//     complexShape.width = complexShape.width / 2 + 1;//TODO this is the shape that is needed in the fftw representation of real valued data in complex space
//     ComplexData result{ this, nullptr, complexShape, shape, complexShape.getVolume() * sizeof(complex_t), 0};
//     IBackendMemoryManager::allocateMemoryOnDevice(result);
//     return result;
// }
//
// ComplexData CUDABackendMemoryManager::allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const{
//     ComplexData result{ this, nullptr, shape, shape, shape.getVolume() * sizeof(complex_t), 0};
//     IBackendMemoryManager::allocateMemoryOnDevice(result);
//     return result;
// }
//

void* CUDABackendMemoryManager::allocateMemoryOnDevice(size_t requested_size) const {
    // Wait for memory if max memory limit is set
    // waitForMemory(requested_size);

    void* devicePtr = nullptr;
    cudaError_t err = cudaMallocAsync(&devicePtr, requested_size, config.stream);

    CUDA_MEMORY_ALLOC_CHECK(err, requested_size, "allocateMemoryOnDevice - cudaMallocAsync", buildCudaContext(config));

    // Synchronize to ensure allocation is complete
    err = cudaStreamSynchronize(config.stream);


    CUDA_CHECK(err, "allocateMemoryOnDevice - cudaStreamSynchronize", buildCudaContext(config));

    // Update memory tracking using getAccess()
    auto access = getMemoryTracking()->getAccess();

    logWithContext(fmt::format("Allocated {:.2f} MB", requested_size / 1e6), LogLevel::DEBUG);
    access.data.totalUsedMemory += requested_size;

    return devicePtr;
}



void* CUDABackendMemoryManager::copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const {
    logWithContext(fmt::format("copyDataToDevice: {:.2f} MB, shape: {}", size / 1e6, shape.print()), LogLevel::DEBUG);
    void* dest = allocateMemoryOnDevice(size);

    cudaError_t err = cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, config.stream);
    CUDA_CHECK(err, "copyDataToDevice - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "copyDataToDevice - cudaStreamSynchronize", buildCudaContext(config));
    return dest;
}

void* CUDABackendMemoryManager::moveDataFromDevice(void* src, size_t size, const CuboidShape& shape, const IBackendMemoryManager& destBackend) const {
    logWithContext(fmt::format("moveDataFromDevice: {:.2f} MB, shape: {}, to {}", size / 1e6, shape.print(), destBackend.getDeviceString()), LogLevel::DEBUG);
    if (&destBackend == this){
        logWithContext(fmt::format("moveDataFromDevice: same backend, returning source pointer directly"), LogLevel::DEBUG);
        return src;
    }
    logWithContext(fmt::format("moveDataFromDevice: cross-backend transfer to {}", destBackend.getDeviceString()), LogLevel::DEBUG);
    void* dest = destBackend.allocateMemoryOnDevice(size);

    cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, config.stream);
    cudaStreamSynchronize(config.stream);

    return dest;
}

void CUDABackendMemoryManager::memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const {
    logWithContext(fmt::format("memCopy: {:.2f} MB, shape: {}", size / 1e6, shape.print()), LogLevel::DEBUG);
    cudaError_t err = cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, config.stream);
    CUDA_CHECK(err, "memCopy - cudaMemcpyAsync", buildCudaContext(config));
    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "memCopy - cudaStreamSynchronize", buildCudaContext(config));
}

void CUDABackendMemoryManager::freeMemoryOnDevice(void* ptr, size_t size) const {
    BACKEND_CHECK(ptr != nullptr, "Attempting to free null pointer", "CUDA", "freeMemoryOnDevice", buildCudaContext(config));
    BACKEND_CHECK(size > 0, "Invalid data size for deallocation", "CUDA", "freeMemoryOnDevice", buildCudaContext(config));

    cudaError_t err = cudaFreeAsync(ptr, config.stream);
    CUDA_CHECK(err, "freeMemoryOnDevice - cudaFreeAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "freeMemoryOnDevice - cudaStreamSynchronize", buildCudaContext(config));

    // Update memory tracking using getAccess()
    auto access = getMemoryTracking()->getAccess();
    if (access.data.totalUsedMemory < size) {
        access.data.totalUsedMemory = static_cast<size_t>(0); // this should never happen
        logWithContext(fmt::format("Memory tracking inconsistency detected in freeMemoryOnDevice"), LogLevel::WARN);
    } else {
        access.data.totalUsedMemory -= size;
    }
    logWithContext(fmt::format("Deallocated {:.2f} MB", size / 1e6), LogLevel::DEBUG);

    ptr = nullptr;
}


size_t CUDABackendMemoryManager::getAvailableMemory() const {
    cudaError_t sync_err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(sync_err, "getAvailableMemory - sync", buildCudaContext(config));

    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    CUDA_CHECK(err, "getAvailableMemory - cudaMemGetInfo", buildCudaContext(config));

    if (freeMem > totalMem) {
        logWithContext(fmt::format("Available memory ({}) exceeds total memory ({})", freeMem, totalMem), LogLevel::WARN);
        return 0; // Return 0 to indicate error condition
    }

    return freeMem;
}

size_t CUDABackendMemoryManager::getAllocatedMemory() const {
    auto access = getMemoryTracking()->getAccess();
    logWithContext(fmt::format("getAllocatedMemory: {:.2f} MB currently allocated", access.data.totalUsedMemory / 1e6), LogLevel::DEBUG);
    return access.data.totalUsedMemory;
}

size_t CUDABackendMemoryManager::estimateFFTWorkspace(const CuboidShape& shape) const {
    int Nx = static_cast<int>(shape.width);
    int Ny = static_cast<int>(shape.height);
    int Nz = static_cast<int>(shape.depth);

    int rank = 3;
    int n[3] = {Nz, Ny, Nx};

    int istride = 1;
    int ostride = 1;

    int inembed_r2c[3] = {Nz, Ny, 2*(Nx/2+1)};
    int onembed_r2c[3] = {Nz, Ny, Nx/2+1};
    int idist_r2c = Nz * Ny * 2*(Nx/2+1);
    int odist_r2c = Nz * Ny * (Nx/2+1);

    size_t r2cWorkSize = 0;
    cufftResult r2cResult = cufftEstimateMany(
        rank, n,
        inembed_r2c, istride, idist_r2c,
        onembed_r2c, ostride, odist_r2c,
        CUFFT_R2C, 1, &r2cWorkSize);

    int inembed_c2r[3] = {Nz, Ny, Nx/2+1};
    int onembed_c2r[3] = {Nz, Ny, 2*(Nx/2+1)};
    int idist_c2r = Nz * Ny * (Nx/2+1);
    int odist_c2r = Nz * Ny * 2*(Nx/2+1);

    size_t c2rWorkSize = 0;
    cufftResult c2rResult = cufftEstimateMany(
        rank, n,
        inembed_c2r, istride, idist_c2r,
        onembed_c2r, ostride, odist_c2r,
        CUFFT_C2R, 1, &c2rWorkSize);

    size_t totalWorkspace = 0;
    if (r2cResult == CUFFT_SUCCESS) totalWorkspace += r2cWorkSize;
    if (c2rResult == CUFFT_SUCCESS) totalWorkspace += c2rWorkSize;

    logWithContext(fmt::format("Estimated cuFFT workspace for shape {}: {:.2f} MB (R2C: {:.2f} MB, C2R: {:.2f} MB)",
        shape.print(), totalWorkspace / 1e6, r2cWorkSize / 1e6, c2rWorkSize / 1e6), LogLevel::DEBUG);

    return totalWorkspace;
}





CUDAComputeBackend::CUDAComputeBackend(CUDABackendConfig config) : config(config){

}

CUDAComputeBackend::~CUDAComputeBackend() {
    destroyPlans();

}

void CUDAComputeBackend::initializePlan(const FFTPlanDescription& description) {
    getPlan(description);
}

cufftHandle CUDAComputeBackend::getPlan(const FFTPlanDescription& description) {

    for (cuFFTPlan& plan : cuFFTPlans){
        if (plan.description == description){
            return plan.plan;
        }
    }

    // Create new plan and store it in the map
    cufftHandle newPlan = initializePlan_(description);
    addPlan(description, newPlan);
    return newPlan;  // Return reference to the stored plan
}

void CUDAComputeBackend::addPlan(const FFTPlanDescription& description, cufftHandle handle){
    cuFFTPlan plan{ handle, description };
    cuFFTPlans.push_back(std::move(plan));
}

// void CUDAComputeBackend::createPlanComplexToReal(cufftHandle& plan, const PlanDescription& description) const {
//     size_t tempSize = sizeof(complex_t) * description.shape.depth * description.shape.height * description.shape.width;
//     CUFFT_CHECK(cufftMakePlan3d(plan, description.shape.depth, description.shape.height, description.shape.width, CUFFT_C2R, &tempSize), "getPlan - C2R plan setup", buildCudaContext(config));
// }

void CUDAComputeBackend::createPlanRealToComplex(cufftHandle& plan, const FFTPlanDescription& description) const {
    int rank = 3;
    long long int Nx = static_cast<long long int>(description.shape.width);
    long long int Ny = static_cast<long long int>(description.shape.height);
    long long int Nz = static_cast<long long int>(description.shape.depth);

    long long int n[3] = {Nz, Ny, Nx};

    // For out-of-place: real input is unpadded, inembed matches logical dimensions.
    // For in-place: real input must be padded on the last dimension to 2*(Nx/2+1)
    //              to accommodate the complex output, so inembed reflects the padded size.
    // onembed is always {Nz, Ny, Nx/2+1} (complex output has halved last dimension).
    long long int inembed[3];
    long long int onembed[3] = {Nz, Ny, Nx/2+1};

    long long int istride = 1;
    long long int ostride = 1;

    long long int idist;
    long long int odist = Nz * Ny * (Nx/2+1);

    // if (description.inPlace) {
    inembed[0] = Nz;
    inembed[1] = Ny;
    inembed[2] = 2*(Nx/2+1);  // padded last dimension (in real_t units)
    idist = Nz * Ny * 2*(Nx/2+1);
    size_t worksize = idist * sizeof(real_t);
    // } else {
    //     inembed[0] = Nz;
    //     inembed[1] = Ny;
    //     inembed[2] = Nx;           // unpadded last dimension
    //     idist = Nz * Ny * Nx;
    // }

    try {

        // Create FFT plan using advanced r2c interface
        CUFFT_RUNTIME_CHECK(cufftMakePlanMany64(
            plan,
            rank, n,
            inembed,
            istride, idist,
            onembed,
            ostride, odist,
            CUFFT_R2C,
            1,
            &worksize
        ), "createPlan - R2C plan setup, might be out of memory", buildCudaContext(config));


        std::string msg = fmt::format(
            "Successfully created cuFFT r2c plan ({}) for shape: {}x{}x{}",
            description.inPlace ? "in-place" : "out-of-place",
            description.shape.width, description.shape.height, description.shape.depth
        );

        logWithContext(msg, LogLevel::DEBUG);

    }
    catch (...) {
        throw;
    }
}

// void CUDAComputeBackend::createPlanRealToComplex(cufftHandle& plan, const PlanDescription& description) const {
//     size_t tempSize = sizeof(complex_t) * description.shape.depth * description.shape.height * description.shape.width;
//     CUFFT_CHECK(cufftMakePlan3d(plan, description.shape.depth, description.shape.height, description.shape.width, CUFFT_R2C, &tempSize), "getPlan - R2C plan setup", buildCudaContext(config));
// }

void CUDAComputeBackend::createPlanComplexToReal(cufftHandle& plan, const FFTPlanDescription& description) const {
    int rank = 3;
    long long int Nx = static_cast<long long int>(description.shape.width);
    long long int Ny = static_cast<long long int>(description.shape.height);
    long long int Nz = static_cast<long long int>(description.shape.depth);

    long long int n[3] = {Nz, Ny, Nx};

    // For out-of-place: real input is unpadded, inembed matches logical dimensions.
    // For in-place: real input must be padded on the last dimension to 2*(Nx/2+1)
    //              to accommodate the complex output, so inembed reflects the padded size.
    // onembed is always {Nz, Ny, Nx/2+1} (complex output has halved last dimension).
    long long int onembed[3];
    long long int inembed[3] = {Nz, Ny, Nx/2+1};

    long long int istride = 1;
    long long int ostride = 1;

    long long int odist;
    long long int idist = Nz * Ny * (Nx/2+1);

    // if (description.inPlace) {
    onembed[0] = Nz;
    onembed[1] = Ny;
    onembed[2] = 2*(Nx/2+1);  // padded last dimension (in real_t units)
    odist = Nz * Ny * 2*(Nx/2+1);
    size_t worksize = odist * sizeof(real_t);
    // } else {
    //     inembed[0] = Nz;
    //     inembed[1] = Ny;
    //     inembed[2] = Nx;           // unpadded last dimension
    //     idist = Nz * Ny * Nx;
    // }

    try {

        // Create FFT plan using advanced r2c interface
        CUFFT_RUNTIME_CHECK(cufftMakePlanMany64(
            plan,
            rank, n,
            inembed,
            istride, idist,
            onembed,
            ostride, odist,
            CUFFT_C2R,
            1,
            &worksize
        ), "createPlan - C2R plan setup, might be out of memory", buildCudaContext(config));


        std::string msg = fmt::format(
            "Successfully created cuFFT c2r plan ({}) for shape: {}x{}x{}",
            description.inPlace ? "in-place" : "out-of-place",
            description.shape.width, description.shape.height, description.shape.depth
        );

        logWithContext(msg, LogLevel::DEBUG);

    }
    catch (...) {
        throw;
    }
}

void CUDAComputeBackend::createPlanComplex(cufftHandle& plan, const FFTPlanDescription& description) const {
    size_t tempSize = sizeof(complex_t) * description.shape.getVolume();
    CUFFT_RUNTIME_CHECK(cufftMakePlan3d(plan, description.shape.depth, description.shape.height, description.shape.width, CUFFT_C2C, &tempSize), "getPlan - C2C plan setup", buildCudaContext(config));
}


cufftHandle CUDAComputeBackend::initializePlan_(const FFTPlanDescription& description){
    // Plan not found, create a new one
    CuboidShape shape = description.shape;

    // Validate input shape
    BACKEND_CHECK(shape.getVolume() > 0, "Invalid shape for FFT plan initialization", "CUDA", "getPlan", buildCudaContext(config));
    BACKEND_CHECK(shape.width > 0 && shape.height > 0 && shape.depth > 0,
                  "Invalid dimensions for FFT plan initialization", "CUDA", "getPlan", buildCudaContext(config));


    cufftHandle newPlan = 0;

    try {
        CUFFT_CHECK(cufftCreate(&newPlan), "getPlan - plan creation", buildCudaContext(config));

        if (description.type == PlanType::COMPLEX) createPlanComplex(newPlan, description);
        else if (description.type == PlanType::REAL && description.direction == PlanDirection::FORWARD) createPlanRealToComplex(newPlan, description);
        else if (description.type == PlanType::REAL && description.direction == PlanDirection::BACKWARD) createPlanComplexToReal(newPlan, description);
        assert(newPlan != 0 && "cufft plan not created");

        CUFFT_CHECK(cufftSetStream(newPlan, config.stream), "getPlan - stream setup", buildCudaContext(config));

        // Synchronize to ensure plan is ready
        cudaError_t err = cudaStreamSynchronize(config.stream);
        CUDA_CHECK(err, "getPlan - cudaStreamSynchronize", buildCudaContext(config));

        return newPlan;
    } catch (...) {
        // Clean up plan if creation fails
        if (newPlan != 0) {
            cufftDestroy(newPlan);
        }
        throw;
    }
}



void CUDAComputeBackend::destroyPlans(){
    for (auto& plan : cuFFTPlans) {
        if (plan.plan != 0) {
            CUFFT_CHECK(cufftDestroy(plan.plan), "destroyFFTPlans - plan destruction", buildCudaContext(config));
            plan.plan = 0;
        }
    }
    cuFFTPlans.clear();
}


// FFT Operations
void CUDAComputeBackend::forwardFFT(const ComplexData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.getSize().getVolume() > 0, "Invalid input data size for forwardFFT", "CUDA", "forwardFFT", buildCudaContext(config));
    BACKEND_CHECK(out.getSize().getVolume() > 0, "Invalid output data size for forwardFFT", "CUDA", "forwardFFT", buildCudaContext(config));
    BACKEND_CHECK(in.getSize().getVolume() == out.getSize().getVolume(), "Size mismatch in forwardFFT", "CUDA", "forwardFFT", buildCudaContext(config));

    // Get or create the forward FFT plan
    FFTPlanDescription desc(PlanDirection::FORWARD, PlanType::COMPLEX, in.getSize(), true);
    cufftHandle forwardPlan = const_cast<CUDAComputeBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(forwardPlan != 0, "Forward FFT plan not initialized", "CUDA", "forwardFFT", buildCudaContext(config));

    CUFFT_CHECK(cufftExecC2C(forwardPlan, reinterpret_cast<cufftComplex*>(in.getData()), reinterpret_cast<cufftComplex*>(out.getData()), CUFFT_FORWARD), "forwardFFT", buildCudaContext(config));
}


// FFT Operations
void CUDAComputeBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.getSize().getVolume() > 0, "Invalid input data size for backwardFFT", "CUDA", "backwardFFT", buildCudaContext(config));
    BACKEND_CHECK(out.getSize().getVolume() > 0, "Invalid output data size for backwardFFT", "CUDA", "backwardFFT", buildCudaContext(config));
    BACKEND_CHECK(in.getSize().getVolume() == out.getSize().getVolume(), "Size mismatch in backwardFFT", "CUDA", "backwardFFT", buildCudaContext(config));

    // Get or create the backward FFT plan
    FFTPlanDescription desc(PlanDirection::BACKWARD, PlanType::COMPLEX, in.getSize(), true);
    cufftHandle backwardPlan = const_cast<CUDAComputeBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(backwardPlan != 0, "Backward FFT plan not initialized", "CUDA", "backwardFFT", buildCudaContext(config));

    CUFFT_CHECK(cufftExecC2C(backwardPlan, reinterpret_cast<cufftComplex*>(in.getData()), reinterpret_cast<cufftComplex*>(out.getData()), CUFFT_INVERSE), "backwardFFT", buildCudaContext(config));

    complex_t normFactor{1.0f / out.getSize().getVolume(), 0.0f};//TESTVALUE
    scalarMultiplication(out, normFactor, out); // Add normalization
}


void CUDAComputeBackend::forwardFFT(const RealData& in, ComplexData& out) const {
    // Validate input data
    BACKEND_CHECK(in.getSize().getVolume() > 0, "Invalid input data size for forwardFFTReal", "CUDA", "forwardFFTReal", buildCudaContext(config));
    BACKEND_CHECK(out.getSize().getVolume() > 0, "Invalid output data size for forwardFFTReal", "CUDA", "forwardFFTReal", buildCudaContext(config));

    // Get or create the forward FFT plan
    FFTPlanDescription desc(PlanDirection::FORWARD, PlanType::REAL, in.getSize(), true);
    cufftHandle forwardPlan = const_cast<CUDAComputeBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(forwardPlan != 0, "forward FFT plan not initialized", "CUDA", "forwardFFTReal", buildCudaContext(config));

    CUFFT_CHECK(cufftExecR2C(forwardPlan, reinterpret_cast<cufftReal*>(in.getData()), reinterpret_cast<cufftComplex*>(out.getData())), "forwardFFTReal", buildCudaContext(config));
}


void CUDAComputeBackend::backwardFFT(const ComplexData& in, RealData& out) const {
    // Validate input data
    BACKEND_CHECK(in.getSize().getVolume() > 0, "Invalid input data size for backwardFFTReal", "CUDA", "backwardFFTReal", buildCudaContext(config));
    BACKEND_CHECK(out.getSize().getVolume() > 0, "Invalid output data size for backwardFFTReal", "CUDA", "backwardFFTReal", buildCudaContext(config));

    // Get or create the backward FFT plan
    FFTPlanDescription desc(PlanDirection::BACKWARD, PlanType::REAL, out.getSize(), true);
    cufftHandle backwardPlan = const_cast<CUDAComputeBackend*>(this)->getPlan(desc);

    // Validate FFT plan
    BACKEND_CHECK(backwardPlan != 0, "Backward FFT plan not initialized", "CUDA", "backwardFFTReal", buildCudaContext(config));

    CUFFT_CHECK(cufftExecC2R(backwardPlan, reinterpret_cast<cufftComplex*>(in.getData()), reinterpret_cast<cufftReal*>(out.getData())), "backwardFFTReal", buildCudaContext(config));

    real_t normFactor{1.0f / out.getSize().getVolume()};//TESTVALUE
    scalarMultiplication(out, normFactor, out); // Add normalization
}


// Shift Operations
void CUDAComputeBackend::octantFourierShift(ComplexData& data) const {
    cudaError_t err = CUBE_FTT::octantFourierShift(static_cast<int>(data.getSize().width), static_cast<int>(data.getSize().height), static_cast<int>(data.getSize().depth), data.getData(), config.stream);
    CUDA_CHECK(err, "octantFourierShift", buildCudaContext(config));
}

void CUDAComputeBackend::octantFourierShift(RealData& data) const {
    int Nx = static_cast<int>(data.getSize().width);
    int stride = Nx + static_cast<int>(data.getPadding());
    cudaError_t err = CUBE_FTT::octantFourierShift(Nx, static_cast<int>(data.getSize().height), static_cast<int>(data.getSize().depth), stride, data.getData(), config.stream);
    CUDA_CHECK(err, "octantFourierShift", buildCudaContext(config));
}

void CUDAComputeBackend::inverseQuadrantShift(ComplexData& data) const {
    // cudaError_t err = CUBE_FTT::octantFourierShift(static_cast<int>(data.getSize().width), static_cast<int>(data.getSize().height), static_cast<int>(data.getSize().depth), data.getData(), config.stream);
    // CUDA_CHECK(err, "octantFourierShift", buildCudaContext(config));
}



void CUDAComputeBackend::sum(const ComplexData& data, complex_t* result) const {
    BACKEND_CHECK(data.getData() != nullptr, "Input data pointer is null", "CUDA", "sum - input data", buildCudaContext(config));
    BACKEND_CHECK(result != nullptr, "Result pointer is null", "CUDA", "sum - result", buildCudaContext(config));

    complex_t* d_result;
    cudaError_t err = cudaMallocAsync(&d_result, sizeof(complex_t), config.stream);
    CUDA_CHECK(err, "sum - cudaMallocAsync", buildCudaContext(config));

    err = cudaMemsetAsync(d_result, 0, sizeof(complex_t), config.stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_result, config.stream);
        CUDA_CHECK(err, "sum - cudaMemsetAsync", buildCudaContext(config));
    }

    err = CUBE_MAT::sum(static_cast<int>(data.getSize().width), static_cast<int>(data.getSize().height), static_cast<int>(data.getSize().depth), data.getData(), d_result, config.stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_result, config.stream);
        CUDA_CHECK(err, "sum", buildCudaContext(config));
    }

    err = cudaMemcpyAsync(result, d_result, sizeof(complex_t), cudaMemcpyDeviceToHost, config.stream);
    cudaFreeAsync(d_result, config.stream);
    CUDA_CHECK(err, "sum - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "sum - cudaStreamSynchronize", buildCudaContext(config));
}

void CUDAComputeBackend::meanSquareError(const ComplexData& a, const ComplexData& b, real_t* result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CUDA", "meanSquareError - input a", buildCudaContext(config));
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CUDA", "meanSquareError - input b", buildCudaContext(config));
    BACKEND_CHECK(result != nullptr, "Result pointer is null", "CUDA", "meanSquareError - result", buildCudaContext(config));

    real_t* d_result;
    cudaError_t err = cudaMallocAsync(&d_result, sizeof(real_t), config.stream);
    CUDA_CHECK(err, "meanSquareError - cudaMallocAsync", buildCudaContext(config));

    err = cudaMemsetAsync(d_result, 0, sizeof(real_t), config.stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_result, config.stream);
        CUDA_CHECK(err, "meanSquareError - cudaMemsetAsync", buildCudaContext(config));
    }

    err = CUBE_MAT::meanSquareError(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), b.getData(), d_result, config.stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_result, config.stream);
        CUDA_CHECK(err, "meanSquareError", buildCudaContext(config));
    }

    real_t d_sumSq;
    err = cudaMemcpyAsync(&d_sumSq, d_result, sizeof(real_t), cudaMemcpyDeviceToHost, config.stream);
    cudaFreeAsync(d_result, config.stream);
    CUDA_CHECK(err, "meanSquareError - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "meanSquareError - cudaStreamSynchronize", buildCudaContext(config));

    *result = d_sumSq / static_cast<real_t>(a.getSize().getVolume());
}

void CUDAComputeBackend::complexAddition(complex_t** dataPointer, ComplexData& sums, int nImages, size_t imageVolume) const {
    cudaError_t err = CUBE_MAT::complexAddition(dataPointer, sums.getData(), nImages, static_cast<int>(imageVolume), config.stream);
}

void CUDAComputeBackend::sumToOne(real_t** data, int nImages, size_t imageVolume) const {
    cudaError_t err = CUBE_MAT::sumToOne(data, nImages, static_cast<int>(imageVolume), config.stream);
}

// Complex Arithmetic Operations
void CUDAComputeBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    // Validate input data
    BACKEND_CHECK(a.getSize().getVolume() > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));
    BACKEND_CHECK(b.getSize().getVolume() > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));
    BACKEND_CHECK(result.getSize().getVolume() > 0, "Invalid output data size for complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));

    cudaError_t err = CUBE_MAT::complexElementwiseMatMul(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), b.getData(), result.getData(), config.stream);
    CUDA_CHECK(err, "complexMultiplication", buildCudaContext(config));
}

void CUDAComputeBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    // Validate input data
    BACKEND_CHECK(a.getSize().getVolume() > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(b.getSize().getVolume() > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(result.getSize().getVolume() > 0, "Invalid output data size for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(epsilon >= 0.0, "Invalid epsilon value for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));

    cudaError_t err = CUBE_MAT::complexElementwiseMatDiv(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), b.getData(), result.getData(), epsilon, config.stream);
    CUDA_CHECK(err, "complexDivision", buildCudaContext(config));
}

void CUDAComputeBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexAddition", "CUDA", "complexAddition", buildCudaContext(config));
    cudaError_t err = CUBE_MAT::complexAddition(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), b.getData() , result.getData(), config.stream);
    CUDA_CHECK(err, "complexAddition", buildCudaContext(config));
}



void CUDAComputeBackend::multiplication(const RealData& a, const RealData& b, RealData& result) const{
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in elementwiseMatMulReal", "CUDA", "elementwiseMatMulReal", buildCudaContext(config));
    int Nx = static_cast<int>(a.getSize().width);
    int strideA = Nx + static_cast<int>(a.getPadding());
    int strideB = static_cast<int>(b.getSize().width + b.getPadding());
    int strideC = static_cast<int>(result.getSize().width + result.getPadding());
    cudaError_t err = CUBE_MAT::elementwiseMatMul(Nx, static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), strideA, strideB, strideC, a.getData(), b.getData(), result.getData(), config.stream);
    CUDA_CHECK(err, "multiplication", buildCudaContext(config));
}
void CUDAComputeBackend::scalarMultiplication(const RealData& a, real_t scalar, RealData& result) const{
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in scalarMultiplicationReal", "CUDA", "scalarMultiplicationReal", buildCudaContext(config));
    int Nx = static_cast<int>(a.getSize().width);
    int strideA = Nx + static_cast<int>(a.getPadding());
    int strideC = static_cast<int>(result.getSize().width + result.getPadding());
    cudaError_t err = CUBE_MAT::scalarMul(Nx, static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), strideA, strideC, a.getData(), scalar , result.getData(), config.stream);
    CUDA_CHECK(err, "scalarMultiplicationReal", buildCudaContext(config));
}
void CUDAComputeBackend::division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const{
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in elementwiseDivisionReal", "CUDA", "elementwiseDivisionReal", buildCudaContext(config));
    int Nx = static_cast<int>(a.getSize().width);
    int strideA = Nx + static_cast<int>(a.getPadding());
    int strideB = static_cast<int>(b.getSize().width + b.getPadding());
    int strideC = static_cast<int>(result.getSize().width + result.getPadding());
    cudaError_t err = CUBE_MAT::elementwiseMatDiv(Nx, static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), strideA, strideB, strideC, a.getData(), b.getData(), result.getData(), epsilon, config.stream);
    CUDA_CHECK(err, "elementwiseDivisionReal", buildCudaContext(config));
}

void CUDAComputeBackend::scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in scalarMultiplication", "CUDA", "scalarMultiplication", buildCudaContext(config));
    cudaError_t err = CUBE_MAT::complexScalarMul(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), scalar[0], scalar[1], result.getData(), config.stream);
    CUDA_CHECK(err, "scalarMultiplication", buildCudaContext(config));
}

void CUDAComputeBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexMultiplicationWithConjugate", "CUDA", "complexMultiplicationWithConjugate", buildCudaContext(config));
    cudaError_t err = CUBE_MAT::complexElementwiseMatMulConjugate(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), b.getData(), result.getData(), config.stream);
    CUDA_CHECK(err, "complexMultiplicationWithConjugate", buildCudaContext(config));
}

void CUDAComputeBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexDivisionStabilized", "CUDA", "complexDivisionStabilized", buildCudaContext(config));
    cudaError_t err = CUBE_MAT::complexElementwiseMatDivStabilized(static_cast<int>(a.getSize().width), static_cast<int>(a.getSize().height), static_cast<int>(a.getSize().depth), a.getData(), b.getData(), result.getData(), epsilon, config.stream);
    CUDA_CHECK(err, "complexDivisionStabilized", buildCudaContext(config));
}




// Specialized Functions
void CUDAComputeBackend::hasNAN(const ComplexData& data) const {
    // Implementation would go here
    logWithContext(fmt::format("hasNAN called on CUDA backend"), LogLevel::DEBUG);
}

// void CUDAComputeBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
//     cudaError_t err = CUBE_REG::calculateLaplacian(psf.getSize().width, psf.getSize().height, psf.getSize().depth, psf.getData(), laplacian.getData(), config.stream);
//     CUDA_CHECK(err, "calculateLaplacianOfPSF", buildCudaContext(config));
// }

// void CUDAComputeBackend::normalizeImage(ComplexData& resultImage, real_t epsilon) const {
//     cudaError_t err = CUBE_FTT::normalizeData(1, 1, 1, resultImage.getData(), config.stream);
//     CUDA_CHECK(err, "normalizeImage", buildCudaContext(config));
// }

// void CUDAComputeBackend::rescaledInverse(ComplexData& data, real_t cubeVolume) const {
//     for (int i = 0; i < data.getSize().getVolume(); ++i) {
//         data.getData()[i][0] /= cubeVolume;
//         data.getData()[i][1] /= cubeVolume;
//     }
// }

// Gradient and TV Functions
void CUDAComputeBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    cudaError_t err1 = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err1, "found it", buildCudaContext(config));

    cudaError_t err = CUBE_REG::gradX(static_cast<int>(image.getSize().width), static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), image.getData(), gradX.getData(), config.stream);
    CUDA_CHECK(err, "gradientX", buildCudaContext(config));
}

void CUDAComputeBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    cudaError_t err = CUBE_REG::gradY(static_cast<int>(image.getSize().width), static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), image.getData(), gradY.getData(), config.stream);
    CUDA_CHECK(err, "gradientY", buildCudaContext(config));
}

void CUDAComputeBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    cudaError_t err = CUBE_REG::gradZ(static_cast<int>(image.getSize().width), static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), image.getData(), gradZ.getData(), config.stream);
    CUDA_CHECK(err, "gradientZ", buildCudaContext(config));
}

void CUDAComputeBackend::computeTV(real_t lambda, const ComplexData& div, ComplexData& tv) const {
    cudaError_t err = CUBE_REG::computeTV(static_cast<int>(div.getSize().width), static_cast<int>(div.getSize().height), static_cast<int>(div.getSize().depth), lambda, div.getData(), tv.getData(), config.stream);
    CUDA_CHECK(err, "computeTV", buildCudaContext(config));
}

void CUDAComputeBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t beta) const {
    cudaError_t err = CUBE_REG::normalizeTV(static_cast<int>(gradX.getSize().width), static_cast<int>(gradX.getSize().height), static_cast<int>(gradX.getSize().depth), gradX.getData(), gradY.getData(), gradZ.getData(), beta, config.stream);
    CUDA_CHECK(err, "normalizeTV", buildCudaContext(config));
}

// Gradient functions for real-valued data
void CUDAComputeBackend::gradientX(const RealData& image, RealData& gradX) const {
    int Nx = static_cast<int>(image.getSize().width);
    int strideIn = Nx + static_cast<int>(image.getPadding());
    int strideOut = static_cast<int>(gradX.getSize().width + gradX.getPadding());
    cudaError_t err = CUBE_REG::gradX(Nx, static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), strideIn, strideOut, image.getData(), gradX.getData(), config.stream);
    CUDA_CHECK(err, "gradientX (real)", buildCudaContext(config));
}

void CUDAComputeBackend::gradientY(const RealData& image, RealData& gradY) const {
    int Nx = static_cast<int>(image.getSize().width);
    int strideIn = Nx + static_cast<int>(image.getPadding());
    int strideOut = static_cast<int>(gradY.getSize().width + gradY.getPadding());
    cudaError_t err = CUBE_REG::gradY(Nx, static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), strideIn, strideOut, image.getData(), gradY.getData(), config.stream);
    CUDA_CHECK(err, "gradientY (real)", buildCudaContext(config));
}

void CUDAComputeBackend::gradientZ(const RealData& image, RealData& gradZ) const {
    int Nx = static_cast<int>(image.getSize().width);
    int strideIn = Nx + static_cast<int>(image.getPadding());
    int strideOut = static_cast<int>(gradZ.getSize().width + gradZ.getPadding());
    cudaError_t err = CUBE_REG::gradZ(Nx, static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), strideIn, strideOut, image.getData(), gradZ.getData(), config.stream);
    CUDA_CHECK(err, "gradientZ (real)", buildCudaContext(config));
}

void CUDAComputeBackend::gradient(const RealData& image, RealData& gradX, RealData& gradY, RealData& gradZ) const {
    int Nx = static_cast<int>(image.getSize().width);
    int strideIn = Nx + static_cast<int>(image.getPadding());
    int strideX = static_cast<int>(gradX.getSize().width + gradX.getPadding());
    int strideY = static_cast<int>(gradY.getSize().width + gradY.getPadding());
    int strideZ = static_cast<int>(gradZ.getSize().width + gradZ.getPadding());
    assert(strideX == strideY && strideY == strideZ);

    cudaError_t err = CUBE_REG::grad(Nx, static_cast<int>(image.getSize().height), static_cast<int>(image.getSize().depth), strideIn, strideX, image.getData(), gradX.getData(), gradY.getData(), gradZ.getData(), config.stream);
    CUDA_CHECK(err, "gradient (real)", buildCudaContext(config));
}

void CUDAComputeBackend::divergence(const RealData& gx, const RealData& gy, const RealData& gz, RealData& result) const {
    int Nx = static_cast<int>(gx.getSize().width);
    int strideGx = Nx + static_cast<int>(gx.getPadding());
    int strideGy = static_cast<int>(gy.getSize().width + gy.getPadding());
    int strideGz = static_cast<int>(gz.getSize().width + gz.getPadding());
    int strideOut = static_cast<int>(result.getSize().width + result.getPadding());
    cudaError_t err = CUBE_REG::divergence(Nx, static_cast<int>(gx.getSize().height), static_cast<int>(gx.getSize().depth), strideGx, strideGy, strideGz, strideOut, gx.getData(), gy.getData(), gz.getData(), result.getData(), config.stream);
    CUDA_CHECK(err, "divergence (real)", buildCudaContext(config));
}

void CUDAComputeBackend::divergence(const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& result) const {
    cudaError_t err = CUBE_REG::divergence(static_cast<int>(gx.getSize().width), static_cast<int>(gx.getSize().height), static_cast<int>(gx.getSize().depth), gx.getData(), gy.getData(), gz.getData(), result.getData(), config.stream);
    CUDA_CHECK(err, "divergence (complex)", buildCudaContext(config));
}

void CUDAComputeBackend::computeTV(real_t lambda, const RealData& div, RealData& tv) const {
    int Nx = static_cast<int>(div.getSize().width);
    int strideDiv = Nx + static_cast<int>(div.getPadding());
    int strideTv = static_cast<int>(tv.getSize().width + tv.getPadding());
    cudaError_t err = CUBE_REG::computeTV(Nx, static_cast<int>(div.getSize().height), static_cast<int>(div.getSize().depth), strideDiv, strideTv, lambda, div.getData(), tv.getData(), config.stream);
    CUDA_CHECK(err, "computeTV (real)", buildCudaContext(config));
}

void CUDAComputeBackend::normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t beta) const {
    int Nx = static_cast<int>(gradX.getSize().width);
    int strideGradX = Nx + static_cast<int>(gradX.getPadding());
    int strideGradY = static_cast<int>(gradY.getSize().width + gradY.getPadding());
    int strideGradZ = static_cast<int>(gradZ.getSize().width + gradZ.getPadding());
    cudaError_t err = CUBE_REG::normalizeTV(Nx, static_cast<int>(gradX.getSize().height), static_cast<int>(gradX.getSize().depth), strideGradX, strideGradY, strideGradZ, gradX.getData(), gradY.getData(), gradZ.getData(), beta, config.stream);
    CUDA_CHECK(err, "normalizeTV (real)", buildCudaContext(config));
}
