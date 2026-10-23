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
#include <cassert>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>



LogCallback g_logger_cuda =[](const std::string& context, const std::string& message, LogLevel level){
    spdlog::info("[{}] {}", context, message);
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
    ensureDevice();
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


RealData CUDABackendMemoryManager::allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const{
    ensureDevice();
    logWithContext(fmt::format("allocateMemoryOnDeviceRealFFTInPlace for shape: {}", shape.print()), LogLevel::LOG_DEBUG);
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
    ensureDevice();
    logWithContext(fmt::format("allocateMemoryOnDeviceReal for shape: {}", shape.print()), LogLevel::LOG_DEBUG);
    std::size_t bytes = shape.getVolume() * sizeof(real_t);

    RealData result{ this, nullptr, shape, shape, bytes, 0};
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}


ComplexData CUDABackendMemoryManager::allocateMemoryOnDeviceComplex(const CuboidShape& shape) const{
    ensureDevice();
    logWithContext(fmt::format("allocateMemoryOnDeviceComplex for shape: {}", shape.print()), LogLevel::LOG_DEBUG);
    CuboidShape complexShape = shape;
    complexShape.width = complexShape.width / 2 + 1;//TODO this is the shape that is needed in the fftw representation of real valued data in complex space
    CuboidShape originalShape = shape;
    ComplexData result{ this, nullptr, complexShape, originalShape, complexShape.getVolume() * sizeof(complex_t), 0};

    // No extra padding: stride equals the complex width
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}
ComplexData CUDABackendMemoryManager::allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const{
    ensureDevice();
    logWithContext(fmt::format("allocateMemoryOnDeviceComplexFull for shape: {}", shape.print()), LogLevel::LOG_DEBUG);
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

    DataView<real_t> result = DataView<real_t>{reinterpret_cast<real_t*>(data.getData()), realShape, realShape, data.getDataBytes(), padding};
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
    DataView<complex_t> result = DataView<complex_t>{reinterpret_cast<complex_t*>(data.getData()), complexShape, data.getSize(), data.getDataBytes(), 0};
    return result;
}


void* CUDABackendMemoryManager::allocateMemoryOnDevice(size_t requested_size) const {
    ensureDevice();
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

    logWithContext(fmt::format("Allocated {:.2f} MB", requested_size / 1e6), LogLevel::LOG_DEBUG);
    access.data.totalUsedMemory += requested_size;

    return devicePtr;
}



void* CUDABackendMemoryManager::copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const {
    ensureDevice();
    logWithContext(fmt::format("copyDataToDevice: {:.2f} MB, shape: {}", size / 1e6, shape.print()), LogLevel::LOG_DEBUG);
    void* dest = allocateMemoryOnDevice(size);

    cudaError_t err = cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, config.stream);
    CUDA_CHECK(err, "copyDataToDevice - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "copyDataToDevice - cudaStreamSynchronize", buildCudaContext(config));
    return dest;
}

void* CUDABackendMemoryManager::moveDataFromDevice(void* src, size_t size, const CuboidShape& shape, const IBackendMemoryManager& destBackend) const {
    ensureDevice();
    logWithContext(fmt::format("moveDataFromDevice: {:.2f} MB, shape: {}, to {}", size / 1e6, shape.print(), destBackend.getDeviceString()), LogLevel::LOG_DEBUG);
    if (&destBackend == this){
        logWithContext(fmt::format("moveDataFromDevice: same backend, returning source pointer directly"), LogLevel::LOG_DEBUG);
        return src;
    }
    logWithContext(fmt::format("moveDataFromDevice: cross-backend transfer to {}", destBackend.getDeviceString()), LogLevel::LOG_DEBUG);
    void* dest = destBackend.allocateMemoryOnDevice(size);

    cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, config.stream);
    cudaStreamSynchronize(config.stream);

    return dest;
}

void CUDABackendMemoryManager::memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const {
    ensureDevice();
    logWithContext(fmt::format("memCopy: {:.2f} MB, shape: {}", size / 1e6, shape.print()), LogLevel::LOG_DEBUG);
    cudaError_t err = cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, config.stream);
    CUDA_CHECK(err, "memCopy - cudaMemcpyAsync", buildCudaContext(config));
    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "memCopy - cudaStreamSynchronize", buildCudaContext(config));
}

void CUDABackendMemoryManager::freeMemoryOnDevice(void* ptr, size_t size) const {
    ensureDevice();
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
        logWithContext(fmt::format("Memory tracking inconsistency detected in freeMemoryOnDevice"), LogLevel::LOG_WARN);
    } else {
        access.data.totalUsedMemory -= size;
    }
    logWithContext(fmt::format("Deallocated {:.2f} MB", size / 1e6), LogLevel::LOG_DEBUG);

    ptr = nullptr;
}


size_t CUDABackendMemoryManager::getAvailableMemory() const {
    ensureDevice();
    cudaError_t sync_err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(sync_err, "getAvailableMemory - sync", buildCudaContext(config));

    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    CUDA_CHECK(err, "getAvailableMemory - cudaMemGetInfo", buildCudaContext(config));

    if (freeMem > totalMem) {
        logWithContext(fmt::format("Available memory ({}) exceeds total memory ({})", freeMem, totalMem), LogLevel::LOG_WARN);
        return 0; // Return 0 to indicate error condition
    }

    return freeMem;
}

size_t CUDABackendMemoryManager::getAllocatedMemory() const {
    auto access = getMemoryTracking()->getAccess();
    logWithContext(fmt::format("getAllocatedMemory: {:.2f} MB currently allocated", access.data.totalUsedMemory / 1e6), LogLevel::LOG_DEBUG);
    return access.data.totalUsedMemory;
}

// estimation not provided for large sizes (>32bit range)
float CUDABackendMemoryManager::estimateFFTWorkspaceCopies(const CuboidShape& shape) const {
    return 2;
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

    inembed[0] = Nz;
    inembed[1] = Ny;
    inembed[2] = 2*(Nx/2+1);  // padded last dimension (in real_t units)
    idist = Nz * Ny * 2*(Nx/2+1);
    size_t worksize = idist * sizeof(real_t);

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

        logWithContext(msg, LogLevel::LOG_DEBUG);
        logWithContext(fmt::format("cuFFT r2c workspace: {:.2f} MB", worksize / 1e6), LogLevel::LOG_DEBUG);

    }
    catch (...) {
        throw;
    }
}

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

    onembed[0] = Nz;
    onembed[1] = Ny;
    onembed[2] = 2*(Nx/2+1);  // padded last dimension (in real_t units)
    odist = Nz * Ny * 2*(Nx/2+1);
    size_t worksize = odist * sizeof(real_t);

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

        logWithContext(msg, LogLevel::LOG_DEBUG);
        logWithContext(fmt::format("cuFFT c2r workspace: {:.2f} MB", worksize / 1e6), LogLevel::LOG_DEBUG);

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
    ensureDevice();
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
    ensureDevice();
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
    ensureDevice();
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
    ensureDevice();
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
    ensureDevice();
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
    ensureDevice();
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
    ensureDevice();
    run("octantFourierShift", CUBE_FTT::octantFourierShift(data.getSize().width, data.getSize().height, data.getSize().depth, data.getData(), config.stream));
}

void CUDAComputeBackend::octantFourierShift(RealData& data) const {
    ensureDevice();
    size_t Nx = data.getSize().width;
    size_t stride = Nx + data.getPadding();
    run("octantFourierShift", CUBE_FTT::octantFourierShift(Nx, data.getSize().height, data.getSize().depth, stride, data.getData(), config.stream));
}

void CUDAComputeBackend::inverseQuadrantShift(ComplexData& data) const {
    ensureDevice();
    // cudaError_t err = CUBE_FTT::octantFourierShift(static_cast<int>(data.getSize().width), static_cast<int>(data.getSize().height), static_cast<int>(data.getSize().depth), data.getData(), config.stream);
    // CUDA_CHECK(err, "octantFourierShift", buildCudaContext(config));
}



void CUDAComputeBackend::sum(const ComplexData& data, complex_t* result) const {
    ensureDevice();
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

    err = CUBE_MAT::sum(data.getSize().width, data.getSize().height, data.getSize().depth, data.getData(), d_result, config.stream);
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
    ensureDevice();
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

    err = CUBE_MAT::meanSquareError(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), b.getData(), d_result, config.stream);
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

void CUDAComputeBackend::complexAddition(complex_t** dataPointer, ComplexData& sums, size_t nImages, size_t imageVolume) const {
    ensureDevice();
    cudaError_t err = CUBE_MAT::complexAddition(dataPointer, sums.getData(), nImages, imageVolume, config.stream);
}

void CUDAComputeBackend::sumToOne(real_t** data, size_t nImages, size_t imageVolume) const {
    ensureDevice();
    cudaError_t err = CUBE_MAT::sumToOne(data, nImages, imageVolume, config.stream);
}

// Complex Arithmetic Operations
void CUDAComputeBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    ensureDevice();
    // Validate input data
    BACKEND_CHECK(a.getSize().getVolume() > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));
    BACKEND_CHECK(b.getSize().getVolume() > 0, "Invalid input data size for complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));
    BACKEND_CHECK(result.getSize().getVolume() > 0, "Invalid output data size for complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexMultiplication", "CUDA", "complexMultiplication", buildCudaContext(config));

    run("complexMultiplication", CUBE_MAT::complexElementwiseMatMul(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), b.getData(), result.getData(), config.stream));
}

void CUDAComputeBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    ensureDevice();
    // Validate input data
    BACKEND_CHECK(a.getSize().getVolume() > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(b.getSize().getVolume() > 0, "Invalid input data size for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(result.getSize().getVolume() > 0, "Invalid output data size for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexDivision", "CUDA", "complexDivision", buildCudaContext(config));
    BACKEND_CHECK(epsilon >= 0.0, "Invalid epsilon value for complexDivision", "CUDA", "complexDivision", buildCudaContext(config));

    run("complexDivision", CUBE_MAT::complexElementwiseMatDiv(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), b.getData(), result.getData(), epsilon, config.stream));
}

void CUDAComputeBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexAddition", "CUDA", "complexAddition", buildCudaContext(config));
    run("complexAddition", CUBE_MAT::complexAddition(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), b.getData() , result.getData(), config.stream));
}



void CUDAComputeBackend::multiplication(const RealData& a, const RealData& b, RealData& result) const{
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in elementwiseMatMulReal", "CUDA", "elementwiseMatMulReal", buildCudaContext(config));
    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideB = b.getSize().width + b.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();
    run("multiplication", CUBE_MAT::elementwiseMatMul(Nx, a.getSize().height, a.getSize().depth, strideA, strideB, strideC, a.getData(), b.getData(), result.getData(), config.stream));
}
void CUDAComputeBackend::scalarMultiplication(const RealData& a, real_t scalar, RealData& result) const{
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in scalarMultiplicationReal", "CUDA", "scalarMultiplicationReal", buildCudaContext(config));
    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();
    run("scalarMultiplicationReal", CUBE_MAT::scalarMul(Nx, a.getSize().height, a.getSize().depth, strideA, strideC, a.getData(), scalar , result.getData(), config.stream));
}
void CUDAComputeBackend::division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const{
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in elementwiseDivisionReal", "CUDA", "elementwiseDivisionReal", buildCudaContext(config));
    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideB = b.getSize().width + b.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();
    run("elementwiseDivisionReal", CUBE_MAT::elementwiseMatDiv(Nx, a.getSize().height, a.getSize().depth, strideA, strideB, strideC, a.getData(), b.getData(), result.getData(), epsilon, config.stream));
}

void CUDAComputeBackend::scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in scalarMultiplication", "CUDA", "scalarMultiplication", buildCudaContext(config));
    run("scalarMultiplication", CUBE_MAT::complexScalarMul(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), scalar[0], scalar[1], result.getData(), config.stream));
}

void CUDAComputeBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexMultiplicationWithConjugate", "CUDA", "complexMultiplicationWithConjugate", buildCudaContext(config));
    run("complexMultiplicationWithConjugate", CUBE_MAT::complexElementwiseMatMulConjugate(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), b.getData(), result.getData(), config.stream));
}

void CUDAComputeBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    ensureDevice();
    BACKEND_CHECK(a.getSize().getVolume() == b.getSize().getVolume() && a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in complexDivisionStabilized", "CUDA", "complexDivisionStabilized", buildCudaContext(config));
    run("complexDivisionStabilized", CUBE_MAT::complexElementwiseMatDivStabilized(a.getSize().width, a.getSize().height, a.getSize().depth, a.getData(), b.getData(), result.getData(), epsilon, config.stream));
}




// Specialized Functions
void CUDAComputeBackend::hasNAN(const ComplexData& data) const {
    ensureDevice();
    // Implementation would go here
    logWithContext(fmt::format("hasNAN called on CUDA backend"), LogLevel::LOG_DEBUG);
}

// Gradient and TV Functions
void CUDAComputeBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    ensureDevice();
    cudaError_t err1 = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err1, "found it", buildCudaContext(config));

    run("gradientX", CUBE_REG::gradX(image.getSize().width, image.getSize().height, image.getSize().depth, image.getData(), gradX.getData(), config.stream));
}

void CUDAComputeBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    ensureDevice();
    run("gradientY", CUBE_REG::gradY(image.getSize().width, image.getSize().height, image.getSize().depth, image.getData(), gradY.getData(), config.stream));
}

void CUDAComputeBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    ensureDevice();
    run("gradientZ", CUBE_REG::gradZ(image.getSize().width, image.getSize().height, image.getSize().depth, image.getData(), gradZ.getData(), config.stream));
}

void CUDAComputeBackend::computeTV(real_t lambda, const ComplexData& div, ComplexData& tv) const {
    ensureDevice();
    run("computeTV", CUBE_REG::computeTV(div.getSize().width, div.getSize().height, div.getSize().depth, lambda, div.getData(), tv.getData(), config.stream));
}

void CUDAComputeBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t beta) const {
    ensureDevice();
    run("normalizeTV", CUBE_REG::normalizeTV(gradX.getSize().width, gradX.getSize().height, gradX.getSize().depth, gradX.getData(), gradY.getData(), gradZ.getData(), beta, config.stream));
}

// Gradient functions for real-valued data
void CUDAComputeBackend::gradientX(const RealData& image, RealData& gradX) const {
    ensureDevice();
    size_t Nx = image.getSize().width;
    size_t strideIn = Nx + image.getPadding();
    size_t strideOut = gradX.getSize().width + gradX.getPadding();
    run("gradientX (real)", CUBE_REG::gradX(Nx, image.getSize().height, image.getSize().depth, strideIn, strideOut, image.getData(), gradX.getData(), config.stream));
}

void CUDAComputeBackend::gradientY(const RealData& image, RealData& gradY) const {
    ensureDevice();
    size_t Nx = image.getSize().width;
    size_t strideIn = Nx + image.getPadding();
    size_t strideOut = gradY.getSize().width + gradY.getPadding();
    run("gradientY (real)", CUBE_REG::gradY(Nx, image.getSize().height, image.getSize().depth, strideIn, strideOut, image.getData(), gradY.getData(), config.stream));
}

void CUDAComputeBackend::gradientZ(const RealData& image, RealData& gradZ) const {
    ensureDevice();
    size_t Nx = image.getSize().width;
    size_t strideIn = Nx + image.getPadding();
    size_t strideOut = gradZ.getSize().width + gradZ.getPadding();
    run("gradientZ (real)", CUBE_REG::gradZ(Nx, image.getSize().height, image.getSize().depth, strideIn, strideOut, image.getData(), gradZ.getData(), config.stream));
}

void CUDAComputeBackend::gradient(const RealData& image, RealData& gradX, RealData& gradY, RealData& gradZ) const {
    ensureDevice();
    size_t Nx = image.getSize().width;
    size_t strideIn = Nx + image.getPadding();
    size_t strideX = gradX.getSize().width + gradX.getPadding();
    size_t strideY = gradY.getSize().width + gradY.getPadding();
    size_t strideZ = gradZ.getSize().width + gradZ.getPadding();
    assert(strideX == strideY && strideY == strideZ);

    run("gradient (real)", CUBE_REG::grad(Nx, image.getSize().height, image.getSize().depth, strideIn, strideX, image.getData(), gradX.getData(), gradY.getData(), gradZ.getData(), config.stream));
}

void CUDAComputeBackend::divergence(const RealData& gx, const RealData& gy, const RealData& gz, RealData& result) const {
    ensureDevice();
    size_t Nx = gx.getSize().width;
    size_t strideGx = Nx + gx.getPadding();
    size_t strideGy = gy.getSize().width + gy.getPadding();
    size_t strideGz = gz.getSize().width + gz.getPadding();
    size_t strideOut = result.getSize().width + result.getPadding();
    run("divergence (real)", CUBE_REG::divergence(Nx, gx.getSize().height, gx.getSize().depth, strideGx, strideGy, strideGz, strideOut, gx.getData(), gy.getData(), gz.getData(), result.getData(), config.stream));
}

void CUDAComputeBackend::divergence(const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& result) const {
    ensureDevice();
    run("divergence (complex)", CUBE_REG::divergence(gx.getSize().width, gx.getSize().height, gx.getSize().depth, gx.getData(), gy.getData(), gz.getData(), result.getData(), config.stream));
}

void CUDAComputeBackend::computeTV(real_t lambda, const RealData& div, RealData& tv) const {
    ensureDevice();
    size_t Nx = div.getSize().width;
    size_t strideDiv = Nx + div.getPadding();
    size_t strideTv = tv.getSize().width + tv.getPadding();
    run("computeTV (real)", CUBE_REG::computeTV(Nx, div.getSize().height, div.getSize().depth, strideDiv, strideTv, lambda, div.getData(), tv.getData(), config.stream));
}

void CUDAComputeBackend::normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t beta) const {
    ensureDevice();
    size_t Nx = gradX.getSize().width;
    size_t strideGradX = Nx + gradX.getPadding();
    size_t strideGradY = gradY.getSize().width + gradY.getPadding();
    size_t strideGradZ = gradZ.getSize().width + gradZ.getPadding();
    run("normalizeTV (real)", CUBE_REG::normalizeTV(Nx, gradX.getSize().height, gradX.getSize().depth, strideGradX, strideGradY, strideGradZ, gradX.getData(), gradY.getData(), gradZ.getData(), beta, config.stream));
}

// ========================================================================
// Real-valued elementwise / reduction primitives (Poisson-likelihood algorithm)
// ========================================================================

void CUDAComputeBackend::realSum(const RealData& a, real_t* result) const {
    ensureDevice();
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CUDA", "realSum - input a", buildCudaContext(config));
    BACKEND_CHECK(result != nullptr, "Result pointer is null", "CUDA", "realSum - result", buildCudaContext(config));

    size_t Nx = a.getSize().width;
    size_t stride = Nx + a.getPadding();

    double* d_result;
    cudaError_t err = cudaMallocAsync(&d_result, sizeof(double), config.stream);
    CUDA_CHECK(err, "realSum - cudaMallocAsync", buildCudaContext(config));

    err = cudaMemsetAsync(d_result, 0, sizeof(double), config.stream);
    CUDA_CHECK(err, "realSum - cudaMemsetAsync", buildCudaContext(config));

    run("realSum", CUBE_MLE::realSum(Nx, a.getSize().height, a.getSize().depth, stride, a.getData(), d_result, config.stream));

    double hostVal = 0.0;
    err = cudaMemcpyAsync(&hostVal, d_result, sizeof(double), cudaMemcpyDeviceToHost, config.stream);
    cudaFreeAsync(d_result, config.stream);
    CUDA_CHECK(err, "realSum - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "realSum - cudaStreamSynchronize", buildCudaContext(config));

    *result = static_cast<real_t>(hostVal);
}

void CUDAComputeBackend::realLinearCombination(const RealData& a, real_t sa, const RealData& b, real_t sb, RealData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CUDA", "realLinearCombination - input a", buildCudaContext(config));
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CUDA", "realLinearCombination - input b", buildCudaContext(config));
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CUDA", "realLinearCombination - result", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in realLinearCombination", "CUDA", "realLinearCombination", buildCudaContext(config));

    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideB = b.getSize().width + b.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();

    run("realLinearCombination", CUBE_MLE::realLinearCombination(Nx, a.getSize().height, a.getSize().depth, strideA, strideB, strideC, a.getData(), sa, b.getData(), sb, result.getData(), config.stream));
}

void CUDAComputeBackend::realAffine(const RealData& a, real_t scalar, real_t add, RealData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CUDA", "realAffine - input a", buildCudaContext(config));
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CUDA", "realAffine - result", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in realAffine", "CUDA", "realAffine", buildCudaContext(config));

    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();

    run("realAffine", CUBE_MLE::realAffine(Nx, a.getSize().height, a.getSize().depth, strideA, strideC, a.getData(), scalar, add, result.getData(), config.stream));
}

void CUDAComputeBackend::realSqrtAffine(const RealData& a, real_t scale, real_t add, real_t lambda, RealData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CUDA", "realSqrtAffine - input a", buildCudaContext(config));
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CUDA", "realSqrtAffine - result", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in realSqrtAffine", "CUDA", "realSqrtAffine", buildCudaContext(config));

    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();

    run("realSqrtAffine", CUBE_MLE::realSqrtAffine(Nx, a.getSize().height, a.getSize().depth, strideA, strideC, a.getData(), scale, add, lambda, result.getData(), config.stream));
}

void CUDAComputeBackend::realClampMin(const RealData& a, real_t lo, RealData& result) const {
    ensureDevice();
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CUDA", "realClampMin - input a", buildCudaContext(config));
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CUDA", "realClampMin - result", buildCudaContext(config));
    BACKEND_CHECK(a.getSize().getVolume() == result.getSize().getVolume(), "Size mismatch in realClampMin", "CUDA", "realClampMin", buildCudaContext(config));

    size_t Nx = a.getSize().width;
    size_t strideA = Nx + a.getPadding();
    size_t strideC = result.getSize().width + result.getPadding();

    run("realClampMin", CUBE_MLE::realClampMin(Nx, a.getSize().height, a.getSize().depth, strideA, strideC, a.getData(), lo, result.getData(), config.stream));
}

void CUDAComputeBackend::poissonLineSearchEval(const RealData& g, const RealData& m, const RealData& hd,
                                               real_t alpha, real_t* J, real_t* dJ, real_t* d2J) const {
    ensureDevice();
    BACKEND_CHECK(g.getData() != nullptr, "Input g pointer is null", "CUDA", "poissonLineSearchEval - input g", buildCudaContext(config));
    BACKEND_CHECK(m.getData() != nullptr, "Input m pointer is null", "CUDA", "poissonLineSearchEval - input m", buildCudaContext(config));
    BACKEND_CHECK(hd.getData() != nullptr, "Input hd pointer is null", "CUDA", "poissonLineSearchEval - input hd", buildCudaContext(config));
    BACKEND_CHECK(J != nullptr && dJ != nullptr && d2J != nullptr, "Result pointers are null", "CUDA", "poissonLineSearchEval - result", buildCudaContext(config));

    size_t Nx = g.getSize().width;
    size_t strideG = Nx + g.getPadding();
    size_t strideM = m.getSize().width + m.getPadding();
    size_t strideHd = hd.getSize().width + hd.getPadding();

    double* d_result;
    cudaError_t err = cudaMallocAsync(&d_result, 3 * sizeof(double), config.stream);
    CUDA_CHECK(err, "poissonLineSearchEval - cudaMallocAsync", buildCudaContext(config));

    err = cudaMemsetAsync(d_result, 0, 3 * sizeof(double), config.stream);
    CUDA_CHECK(err, "poissonLineSearchEval - cudaMemsetAsync", buildCudaContext(config));

    run("poissonLineSearchEval", CUBE_MLE::poissonLineSearchEval(Nx, g.getSize().height, g.getSize().depth, strideG, strideM, strideHd, g.getData(), m.getData(), hd.getData(), alpha, d_result, config.stream));

    double hostResult[3] = {0.0, 0.0, 0.0};
    err = cudaMemcpyAsync(hostResult, d_result, 3 * sizeof(double), cudaMemcpyDeviceToHost, config.stream);
    cudaFreeAsync(d_result, config.stream);
    CUDA_CHECK(err, "poissonLineSearchEval - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "poissonLineSearchEval - cudaStreamSynchronize", buildCudaContext(config));

    *J = static_cast<real_t>(hostResult[0]);
    *dJ = static_cast<real_t>(hostResult[1]);
    *d2J = static_cast<real_t>(hostResult[2]);
}

void CUDAComputeBackend::fillGaussianOTF(ComplexData& out, real_t sigmaX, real_t sigmaY, real_t sigmaZ) const {
    ensureDevice();
    size_t Nx = out.getSize().width;
    size_t strideGradX = Nx + out.getPadding();

    // Gaussian OTF in cycles per sample: OTF(f) = exp(-2*pi^2 * sigma^2 * f^2),
    // i.e. the Fourier transform of a spatial Gaussian with std-dev `sigma` samples.
    // (Do NOT use 2*(2*pi)^2*sigma^2 = 8*pi^2*sigma^2: that is the angular-frequency
    //  form and is 4x too large for cycles/sample bins.)
    const real_t twoPiSq = (real_t)M_PI * (real_t)M_PI * 2.0f;
    const real_t ax = twoPiSq * sigmaX * sigmaX;
    const real_t ay = twoPiSq * sigmaY * sigmaY;
    const real_t az = twoPiSq * sigmaZ * sigmaZ;
    run("fillGaussianOTF", CUBE_MLE::fillGaussianOTF(Nx, out.getSize().height, out.getSize().depth, strideGradX,  out.getData(), twoPiSq, ax, ay, az, config.stream));
}

void CUDAComputeBackend::normalizePsfUnitSum(ComplexData& psf, real_t epsilon) const {
    ensureDevice();
    BACKEND_CHECK(psf.getData() != nullptr, "Input psf pointer is null", "CUDA", "normalizePsfUnitSum - input psf", buildCudaContext(config));

    // The DC bin (element [0] of the half-complex buffer) is device memory, so read
    // back just the 8-byte DC element to the host before scaling.
    real_t hostDc[2] = {0.0f, 0.0f};
    cudaError_t err = cudaMemcpyAsync(hostDc, psf.getData(), sizeof(hostDc), cudaMemcpyDeviceToHost, config.stream);
    CUDA_CHECK(err, "normalizePsfUnitSum - cudaMemcpyAsync", buildCudaContext(config));

    err = cudaStreamSynchronize(config.stream);
    CUDA_CHECK(err, "normalizePsfUnitSum - cudaStreamSynchronize", buildCudaContext(config));

    const real_t dc = hostDc[0];
    if (std::abs(dc) <= epsilon) return;

    // scalarMultiplication takes complex_t (real_t[2]) by value: pass a named
    // array, not a braced temporary (which would decay to a pointer).
    complex_t scale = {1.0f / dc, 0.0f};
    scalarMultiplication(psf, scale, psf);
}
