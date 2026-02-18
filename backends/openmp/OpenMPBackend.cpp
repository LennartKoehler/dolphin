#include "OpenMPBackend.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstring>
#include <cassert>
#include "dolphinbackend/Exceptions.h"
#ifdef __linux__
#include <unistd.h>
#endif

#include <omp.h>

// Include  headers for high-performance implementations
#ifdef __AVX2__
#include <immintrin.h>
#define _ALIGNMENT 32
#elif defined(__SSE2__)
#include <emmintrin.h>
#define _ALIGNMENT 16
#else
#define _ALIGNMENT 8
#endif

// Unified FFTW error check macro
#define FFTW_UNIFIED_CHECK(fftwf_result, operation) { \
    if ((fftwf_result) == nullptr) { \
        throw dolphin::backend::BackendException( \
            "FFTW operation failed", \
            "OpenMP", \
            operation \
        ); \
    } \
}
// Unified FFTW malloc error check macro
#define FFTW_MALLOC_UNIFIED_CHECK(ptr, size, operation) { \
    if ((ptr) == nullptr) { \
        throw dolphin::backend::MemoryException( \
            "FFTW memory allocation failed", \
            "OpenMP", \
            size, \
            operation \
        ); \
    } \
}

// Logger support
static LogCallback g_logger;

void set_backend_logger(LogCallback cb) {
    g_logger = cb;
}

// Static member definition
MemoryTracking OpenMPBackendMemoryManager::memory;

extern "C" IDeconvolutionBackend* createDeconvolutionBackend() {
    return new OpenMPDeconvolutionBackend();
}

extern "C" IBackendMemoryManager* createBackendMemoryManager() {
    return new OpenMPBackendMemoryManager();
}

extern "C" IBackend* createBackend(){
    return OpenMPBackend::create();
}

// OpenMPBackendMemoryManager implementation
OpenMPBackendMemoryManager::OpenMPBackendMemoryManager(){
    // Initialize memory tracking if not already done
    std::unique_lock<std::mutex> lock(memory.memoryMutex);
    if (memory.maxMemorySize == 0) {
        memory.maxMemorySize = getAvailableMemory();
    }
}

OpenMPBackendMemoryManager::~OpenMPBackendMemoryManager() {

}

void OpenMPBackendMemoryManager::setMemoryLimit(size_t maxMemorySize) {
    std::unique_lock<std::mutex> lock(memory.memoryMutex);
    memory.maxMemorySize = maxMemorySize;
}

void OpenMPBackendMemoryManager::waitForMemory(size_t requiredSize) const {
    std::unique_lock<std::mutex> lock(memory.memoryMutex);
    if ((memory.totalUsedMemory + requiredSize) > memory.maxMemorySize){
        throw dolphin::backend::MemoryException("Exceeded set memory constraint", "OpenMP", requiredSize, "Memory Allocation");
    }
}

// OpenMPBackendMemoryManager implementation
bool OpenMPBackendMemoryManager::isOnDevice(void* ptr) const {
    // For OpenMP backend, all valid pointers are "on device"
    return ptr != nullptr;
}

void OpenMPBackendMemoryManager::allocateMemoryOnDevice(ComplexData& data) const {
    if (data.data != nullptr) {
        return; // Already allocated
    }
    
    size_t requested_size = sizeof(complex_t) * data.size.getVolume();
    
    // Wait for memory if max memory limit is set
    waitForMemory(requested_size);
    
    data.data = (complex_t*)fftwf_malloc(requested_size);
    MEMORY_ALLOC_CHECK(data.data, requested_size, "OpenMP", "allocateMemoryOnDevice");
    
    // Update memory tracking
    {
        std::unique_lock<std::mutex> lock(memory.memoryMutex);
        memory.totalUsedMemory += requested_size;
    }
    
    data.backend = this;
}

ComplexData OpenMPBackendMemoryManager::allocateMemoryOnDevice(const CuboidShape& shape) const {
    ComplexData result{this, nullptr, shape};
    allocateMemoryOnDevice(result);
    return result;
}

ComplexData OpenMPBackendMemoryManager::copyDataToDevice(const ComplexData& srcdata) const {
    BACKEND_CHECK(srcdata.data != nullptr, "Source data pointer is null", "OpenMP", "copyDataToDevice - source data");
    ComplexData result = allocateMemoryOnDevice(srcdata.size);
    std::memcpy(result.data, srcdata.data, srcdata.size.getVolume() * sizeof(complex_t));
    return result;
}

ComplexData OpenMPBackendMemoryManager::moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const {
    BACKEND_CHECK(srcdata.data != nullptr, "Source data pointer is null", "OpenMP", "moveDataFromDevice - source data");
    if (&destBackend == this){
        return srcdata;
    }
    else{
        // For cross-backend transfer, use the destination backend's copy method
        return destBackend.copyDataToDevice(srcdata);
    }
}

ComplexData OpenMPBackendMemoryManager::copyData(const ComplexData& srcdata) const {
    BACKEND_CHECK(srcdata.data != nullptr, "Source data pointer is null", "OpenMP", "copyData - source data");
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    memCopy(srcdata, destdata);
    return destdata;
}



void OpenMPBackendMemoryManager::memCopy(const ComplexData& srcData, ComplexData& destData) const {
    BACKEND_CHECK(srcData.data != nullptr, "Source data pointer is null", "OpenMP", "memCopy - source data");
    BACKEND_CHECK(destData.data != nullptr, "Destination data pointer is null", "OpenMP", "memCopy - destination data");
    BACKEND_CHECK(destData.size.getVolume() == srcData.size.getVolume(), "Source and destination must have same size", "OpenMP", "memCopy");
    std::memcpy(destData.data, srcData.data, srcData.size.getVolume() * sizeof(complex_t));
}

void OpenMPBackendMemoryManager::freeMemoryOnDevice(ComplexData& data) const {
    BACKEND_CHECK(data.data != nullptr, "Data pointer is null", "OpenMP", "freeMemoryOnDevice - data pointer");
    size_t requested_size = sizeof(complex_t) * data.size.getVolume();
    fftwf_free(data.data);
    
    // Update memory tracking
    {
        std::unique_lock<std::mutex> lock(memory.memoryMutex);
        if (memory.totalUsedMemory < requested_size) {
            memory.totalUsedMemory = static_cast<size_t>(0); // this should never happen
        } else {
            memory.totalUsedMemory -= requested_size;
        }
        // Notify waiting threads that memory is now available
        memory.memoryCondition.notify_all();
    }
    
    data.data = nullptr;
}

size_t OpenMPBackendMemoryManager::getAvailableMemory() const {
    try {
        // For OpenMP backend, return available system memory
        size_t memory = 0;
        #ifdef __linux__
            long pagesize = sysconf(_SC_PAGESIZE);
            long pages = sysconf(_SC_AVPHYS_PAGES);
            if (pagesize > 0 && pages > 0) {
                memory = static_cast<size_t>(pagesize) * static_cast<size_t>(pages);
            }
        #elif _WIN32
            #include <windows.h>
            MEMORYSTATUSEX status;
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            memory = static_cast<size_t>(status.ullAvailPhys);
        #endif
        
        if (memory == 0) {
            throw std::runtime_error("Failed to get available memory");
        }
        return memory;
    } catch (const std::exception& e) {
        if (g_logger) {
            g_logger(std::format("Exception in getAvailableMemory: {}", e.what()), LogLevel::ERROR);
        }
        throw; // Re-throw to propagate the exception
    }
}

size_t OpenMPBackendMemoryManager::getAllocatedMemory() const {
    std::lock_guard<std::mutex> lock(memory.memoryMutex);
    return memory.totalUsedMemory;
}


// #####################################################################################################
// OpenMPDeconvolutionBackend implementation
OpenMPDeconvolutionBackend::OpenMPDeconvolutionBackend() {
}

OpenMPDeconvolutionBackend::~OpenMPDeconvolutionBackend() {
    destroyFFTPlans();
}

void OpenMPDeconvolutionBackend::initializeGlobal() {
    static bool globalInitialized = false;
    if (!globalInitialized){
        fftwf_init_threads();
        //TODO how does this work which is globale tec. how do i do this with normal cpubackend, they kind of init the same fftw each?
        int num_threads = nThreads < omp_get_max_threads() ? nThreads : omp_get_max_threads();
        omp_set_num_threads(num_threads);
        fftwf_plan_with_nthreads(num_threads);
        if (g_logger) {
            g_logger(std::format("OpenMP global initialization done with {} threads", num_threads), LogLevel::DEBUG);
        }
        globalInitialized = true;
    }
}

void OpenMPDeconvolutionBackend::init(const BackendConfig& config) {
    initializeGlobal();
    if (g_logger) {
        g_logger("OpenMP backend initialized for lazy plan creation", LogLevel::DEBUG);
    }
}

void OpenMPDeconvolutionBackend::cleanup() {
    fftwf_cleanup_threads();
    destroyFFTPlans();
    if (g_logger) {
        g_logger("OpenMP backend postprocessing completed", LogLevel::DEBUG);
    }
}

void OpenMPDeconvolutionBackend::initializePlan(const CuboidShape& shape) {
    // This method assumes the mutex is already locked by the caller
    
    // Check if plan already exists for this shape (real_t-check pattern)
    if (planMap.find(shape) != planMap.end()) {
        return; // Plan already exists
    }
    
    // Allocate temporary memory for plan creation
    complex_t* temp = nullptr;
    try{
        temp = (complex_t*)fftwf_malloc(sizeof(complex_t) * shape.getVolume());
        FFTW_MALLOC_UNIFIED_CHECK(temp, sizeof(complex_t) * shape.getVolume(), "initializePlan");
        
        FFTPlanPair& planPair = planMap[shape];
        
        // Create forward FFT plan with OpenMP threading
        planPair.forward = fftwf_plan_dft_3d(shape.depth, shape.height, shape.width,
                                           temp, temp, FFTW_FORWARD, FFTW_MEASURE);
        FFTW_UNIFIED_CHECK(planPair.forward, "initializePlan - forward plan");

        g_logger(std::string("FFTWF3 plan using ") + fftwf_sprint_plan(planPair.forward), LogLevel::DEBUG);
        
        // Create backward FFT plan with OpenMP threading  
        planPair.backward = fftwf_plan_dft_3d(shape.depth, shape.height, shape.width,
                                            temp, temp, FFTW_BACKWARD, FFTW_MEASURE);
        FFTW_UNIFIED_CHECK(planPair.backward, "initializePlan - backward plan");
        
        fftwf_free(temp);
        
        if (g_logger) {
            std::string msg = std::format(
                "Successfully created FFTW plan for shape: {}x{}x{}",
                shape.width, shape.height, shape.depth
            );
            g_logger(msg, LogLevel::INFO);
        }
    }
    catch (...){
        if (temp != nullptr){
            fftwf_free(temp);
        }
        throw;
    }
}

void OpenMPDeconvolutionBackend::destroyFFTPlans() {
    std::unique_lock<std::mutex> lock(backendMutex);
    for (auto& pair : planMap) {
        if (pair.second.forward) {
            fftwf_destroy_plan(pair.second.forward);
            pair.second.forward = nullptr;
        }
        if (pair.second.backward) {
            fftwf_destroy_plan(pair.second.backward);
            pair.second.backward = nullptr;
        }
    }
    planMap.clear();
}

OpenMPDeconvolutionBackend::FFTPlanPair* OpenMPDeconvolutionBackend::getPlanPair(const CuboidShape& shape) {
    auto it = planMap.find(shape);
    if (it != planMap.end()) {
        return &it->second;
    }
    return nullptr;
}

// FFT Operations
void OpenMPDeconvolutionBackend::forwardFFT(const ComplexData& in, ComplexData& out) const {
    BACKEND_CHECK(in.data != nullptr, "Input data pointer is null", "OpenMP", "forwardFFT - input data");
    BACKEND_CHECK(out.data != nullptr, "Output data pointer is null", "OpenMP", "forwardFFT - output data");
    
    // First, try to get existing plan (fast path, no lock)
    auto* planPair = const_cast<OpenMPDeconvolutionBackend*>(this)->getPlanPair(in.size);
    
    // If plan doesn't exist, create it (slow path, with lock)
    if (planPair == nullptr) {
        std::unique_lock<std::mutex> lock(const_cast<OpenMPDeconvolutionBackend*>(this)->backendMutex);
        // real_t-check pattern: another thread might have created it while we waited for the lock
        planPair = const_cast<OpenMPDeconvolutionBackend*>(this)->getPlanPair(in.size);
        if (planPair == nullptr) {
            const_cast<OpenMPDeconvolutionBackend*>(this)->initializePlan(in.size);
            planPair = const_cast<OpenMPDeconvolutionBackend*>(this)->getPlanPair(in.size);
        }
    }
    
    BACKEND_CHECK(planPair != nullptr, "Failed to create FFT plan for shape", "OpenMP", "forwardFFT - plan creation");
    BACKEND_CHECK(planPair->forward != nullptr, "Forward FFT plan is null", "OpenMP", "forwardFFT - FFT plan");
    
    fftwf_execute_dft(planPair->forward, reinterpret_cast<fftwf_complex*>(in.data), reinterpret_cast<fftwf_complex*>(out.data));
}

void OpenMPDeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    BACKEND_CHECK(in.data != nullptr, "Input data pointer is null", "OpenMP", "backwardFFT - input data");
    BACKEND_CHECK(out.data != nullptr, "Output data pointer is null", "OpenMP", "backwardFFT - output data");
    
    // First, try to get existing plan (fast path, no lock)
    auto* planPair = const_cast<OpenMPDeconvolutionBackend*>(this)->getPlanPair(in.size);
    
    // If plan doesn't exist, create it (slow path, with lock)
    if (planPair == nullptr) {
        std::unique_lock<std::mutex> lock(const_cast<OpenMPDeconvolutionBackend*>(this)->backendMutex);
        // real_t-check pattern: another thread might have created it while we waited for the lock
        planPair = const_cast<OpenMPDeconvolutionBackend*>(this)->getPlanPair(in.size);
        if (planPair == nullptr) {
            const_cast<OpenMPDeconvolutionBackend*>(this)->initializePlan(in.size);
            planPair = const_cast<OpenMPDeconvolutionBackend*>(this)->getPlanPair(in.size);
        }
    }
    
    BACKEND_CHECK(planPair != nullptr, "Failed to create FFT plan for shape", "OpenMP", "backwardFFT - plan creation");
    BACKEND_CHECK(planPair->backward != nullptr, "Backward FFT plan is null", "OpenMP", "backwardFFT - FFT plan");
    
    fftwf_execute_dft(planPair->backward, reinterpret_cast<fftwf_complex*>(in.data), reinterpret_cast<fftwf_complex*>(out.data));
}

// Shift Operations
void OpenMPDeconvolutionBackend::octantFourierShift(ComplexData& data) const {
    int width = data.size.width;
    int height = data.size.height;
    int depth = data.size.depth;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = ((z + halfDepth) % depth) * height * width +
                          ((y + halfHeight) % height) * width +
                          ((x + halfWidth) % width);

                if (idx1 != idx2) {
                    std::swap(data.data[idx1][0], data.data[idx2][0]);
                    std::swap(data.data[idx1][1], data.data[idx2][1]);
                }
            }
        }
    }
}

void OpenMPDeconvolutionBackend::inverseQuadrantShift(ComplexData& data) const {
    int width = data.size.width;
    int height = data.size.height;
    int depth = data.size.depth;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    #pragma omp parallel for num_threads(nThreads) collapse(3)
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

    #pragma omp parallel for num_threads(nThreads) collapse(3)
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

    #pragma omp parallel for num_threads(nThreads) collapse(3)
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

    #pragma omp parallel for num_threads(nThreads) collapse(3)
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
void OpenMPDeconvolutionBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "complexMultiplication - input a");
    BACKEND_CHECK(b.data != nullptr, "Input b pointer is null", "OpenMP", "complexMultiplication - input b");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "complexMultiplication - result");
    
    const int volume = a.size.getVolume();
    
    // Use restrict pointers to help compiler optimize
    complex_t* __restrict__ a_ptr = a.data;
    complex_t* __restrict__ b_ptr = b.data;
    complex_t* __restrict__ result_ptr = result.data;
    
    #pragma omp parallel for num_threads(nThreads)  //aligned(a_ptr, b_ptr, result_ptr:SIMD_ALIGNMENT) schedule(static)
    for (int i = 0; i < volume; ++i) {
        // Direct access without intermediate variables for better vectorization
        const real_t real_a = a_ptr[i][0];
        const real_t imag_a = a_ptr[i][1];
        const real_t real_b = b_ptr[i][0];
        const real_t imag_b = b_ptr[i][1];

        result_ptr[i][0] = real_a * real_b - imag_a * imag_b;
        result_ptr[i][1] = real_a * imag_b + imag_a * real_b;
    }
}




void OpenMPDeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "complexDivision - input a");
    BACKEND_CHECK(b.data != nullptr, "Input b pointer is null", "OpenMP", "complexDivision - input b");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "complexDivision - result");

    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < a.size.getVolume(); ++i) {
        real_t real_a = a.data[i][0];
        real_t imag_a = a.data[i][1];
        real_t real_b = b.data[i][0];
        real_t imag_b = b.data[i][1];

        real_t denominator = real_b * real_b + imag_b * imag_b;

        if (denominator < epsilon) {
            result.data[i][0] = 0.0;
            result.data[i][1] = 0.0;
        } else {
            result.data[i][0] = (real_a * real_b + imag_a * imag_b) / denominator;
            result.data[i][1] = (imag_a * real_b - real_a * imag_b) / denominator;
        }
    }
}

void OpenMPDeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "complexAddition - input a");
    BACKEND_CHECK(b.data != nullptr, "Input b pointer is null", "OpenMP", "complexAddition - input b");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "complexAddition - result");

    const int volume = a.size.getVolume();
    complex_t* __restrict__ a_ptr = a.data;
    complex_t* __restrict__ b_ptr = b.data;
    complex_t* __restrict__ result_ptr = result.data;

    #pragma omp parallel for num_threads(nThreads)  //aligned(a_ptr, b_ptr, result_ptr:SIMD_ALIGNMENT) schedule(static)
    for (int i = 0; i < volume; ++i) {
        result_ptr[i][0] = a_ptr[i][0] + b_ptr[i][0];
        result_ptr[i][1] = a_ptr[i][1] + b_ptr[i][1];
    }
}

void OpenMPDeconvolutionBackend::scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "scalarMultiplication - input a");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "scalarMultiplication - result");

    const int volume = a.size.getVolume();
    complex_t* __restrict__ a_ptr = a.data;
    complex_t* __restrict__ result_ptr = result.data;
    real_t scalar_real = scalar[0];
    real_t scalar_imag = scalar[1];

    #pragma omp parallel for num_threads(nThreads)//  aligned(a_ptr, result_ptr:SIMD_ALIGNMENT) schedule(static)
    for (int i = 0; i < volume; ++i) {
        result_ptr[i][0] = a_ptr[i][0] * scalar_real;
        result_ptr[i][1] = a_ptr[i][1] * scalar_imag;
    }
}

void OpenMPDeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "complexMultiplicationWithConjugate - input a");
    BACKEND_CHECK(b.data != nullptr, "Input b pointer is null", "OpenMP", "complexMultiplicationWithConjugate - input b");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "complexMultiplicationWithConjugate - result");

    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < a.size.getVolume(); ++i) {
        real_t real_a = a.data[i][0];
        real_t imag_a = a.data[i][1];
        real_t real_b = b.data[i][0];
        real_t imag_b = -b.data[i][1];  // Conjugate

        result.data[i][0] = real_a * real_b - imag_a * imag_b;
        result.data[i][1] = real_a * imag_b + imag_a * real_b;
    }
}

void OpenMPDeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "complexDivisionStabilized - input a");
    BACKEND_CHECK(b.data != nullptr, "Input b pointer is null", "OpenMP", "complexDivisionStabilized - input b");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "complexDivisionStabilized - result");

    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < a.size.getVolume(); ++i) {
        real_t real_a = a.data[i][0];
        real_t imag_a = a.data[i][1];
        real_t real_b = b.data[i][0];
        real_t imag_b = b.data[i][1];

        real_t mag = std::max(epsilon, real_b * real_b + imag_b * imag_b);

        result.data[i][0] = (real_a * real_b + imag_a * imag_b) / mag;
        result.data[i][1] = (imag_a * real_b - real_a * imag_b) / mag;
    }
}

// Specialized Functions
void OpenMPDeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
    int width = psf.size.width;
    int height = psf.size.height;
    int depth = psf.size.depth;

    #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                real_t wz = 2 * M_PI * z / depth;
                real_t wy = 2 * M_PI * y / height;
                real_t wx = 2 * M_PI * x / width;
                real_t laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

                int index = (z * height + y) * width + x;

                laplacian.data[index][0] = psf.data[index][0] * laplacian_value;
                laplacian.data[index][1] = psf.data[index][1] * laplacian_value;
            }
        }
    }
}

void OpenMPDeconvolutionBackend::normalizeImage(ComplexData& resultImage, real_t epsilon) const {
    real_t max_val = 0.0, max_val2 = 0.0;
    
    #pragma omp parallel for num_threads(nThreads) reduction(max:max_val, max_val2)
    for (int j = 0; j < resultImage.size.getVolume(); j++) {
        max_val = std::max(max_val, resultImage.data[j][0]);
        max_val2 = std::max(max_val2, resultImage.data[j][1]);
    }
    
    #pragma omp parallel for num_threads(nThreads) 
    for (int j = 0; j < resultImage.size.getVolume(); j++) {
        resultImage.data[j][0] /= (max_val + epsilon);
        resultImage.data[j][1] /= (max_val2 + epsilon);
    }
}

void OpenMPDeconvolutionBackend::rescaledInverse(ComplexData& data, real_t cubeVolume) const {
    #pragma omp parallel for num_threads(nThreads) 
    for (int i = 0; i < data.size.getVolume(); ++i) {
        data.data[i][0] /= cubeVolume;
        data.data[i][1] /= cubeVolume;
    }
}

// Debug functions
void OpenMPDeconvolutionBackend::hasNAN(const ComplexData& data) const {

}

// Layer and Visualization Functions
void OpenMPDeconvolutionBackend::reorderLayers(ComplexData& data) const {
    int width = data.size.width;
    int height = data.size.height;
    int depth = data.size.depth;
    int layerSize = width * height;
    int halfDepth = depth / 2;
    
    complex_t* temp = (complex_t*)fftwf_malloc(sizeof(complex_t) * data.size.getVolume());
    FFTW_MALLOC_UNIFIED_CHECK(temp, sizeof(complex_t) * data.size.getVolume(), "reorderLayers");

    int destIndex = 0;

    // Copy the middle layer to the first position
    std::memcpy(temp + destIndex * layerSize, data.data + halfDepth * layerSize, sizeof(complex_t) * layerSize);
    destIndex++;

    // Copy the layers after the middle layer
    for (int z = halfDepth + 1; z < depth; ++z) {
        std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(complex_t) * layerSize);
        destIndex++;
    }

    // Copy the layers before the middle layer
    for (int z = 0; z < halfDepth; ++z) {
        std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(complex_t) * layerSize);
        destIndex++;
    }

    // Copy reordered data back to the original array
    std::memcpy(data.data, temp, sizeof(complex_t) * data.size.getVolume());
    fftwf_free(temp);
}

// Gradient and TV Functions
void OpenMPDeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    int width = image.size.width;
    int height = image.size.height;
    int depth = image.size.depth;
    
    #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                
                if (x < width - 1) {
                    int nextIndex = index + 1;
                    gradX.data[index][0] = image.data[index][0] - image.data[nextIndex][0];
                    gradX.data[index][1] = image.data[index][1] - image.data[nextIndex][1];
                } else {
                    // Boundary condition: last column
                    gradX.data[index][0] = 0.0;
                    gradX.data[index][1] = 0.0;
                }
            }
        }
    }
}

void OpenMPDeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    int width = image.size.width;
    int height = image.size.height;
    int depth = image.size.depth;
    
    #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                
                if (y < height - 1) {
                    int nextIndex = index + width;
                    gradY.data[index][0] = image.data[index][0] - image.data[nextIndex][0];
                    gradY.data[index][1] = image.data[index][1] - image.data[nextIndex][1];
                } else {
                    // Boundary condition: last row
                    gradY.data[index][0] = 0.0;
                    gradY.data[index][1] = 0.0;
                }
            }
        }
    }
}

void OpenMPDeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    int width = image.size.width;
    int height = image.size.height;
    int depth = image.size.depth;
    
    #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                
                if (z < depth - 1) {
                    int nextIndex = index + height * width;
                    gradZ.data[index][0] = image.data[index][0] - image.data[nextIndex][0];
                    gradZ.data[index][1] = image.data[index][1] - image.data[nextIndex][1];
                } else {
                    // Boundary condition: last depth layer
                    gradZ.data[index][0] = 0.0;
                    gradZ.data[index][1] = 0.0;
                }
            }
        }
    }
}

void OpenMPDeconvolutionBackend::computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
    int nxy = gx.size.width * gx.size.height;

    #pragma omp parallel for num_threads(nThreads)
    for (int z = 0; z < gx.size.depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            real_t dx = gx.data[index][0];
            real_t dy = gy.data[index][0];
            real_t dz = gz.data[index][0];

            tv.data[index][0] = 1.0 / ((dx + dy + dz) * lambda + 1.0);
            tv.data[index][1] = 0.0;
        }
    }
}



void OpenMPDeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const {
    int nxy = gradX.size.width * gradX.size.height;

    #pragma omp parallel for num_threads(nThreads)
    for (int z = 0; z < gradX.size.depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            real_t norm = std::sqrt(
                gradX.data[index][0] * gradX.data[index][0] + gradX.data[index][1] * gradX.data[index][1] +
                gradY.data[index][0] * gradY.data[index][0] + gradY.data[index][1] * gradY.data[index][1] +
                gradZ.data[index][0] * gradZ.data[index][0] + gradZ.data[index][1] * gradZ.data[index][1]
            );

            norm = std::max(norm, epsilon);

            gradX.data[index][0] /= norm;
            gradX.data[index][1] /= norm;
            gradY.data[index][0] /= norm;
            gradY.data[index][1] /= norm;
            gradZ.data[index][0] /= norm;
            gradZ.data[index][1] /= norm;
        }
    }
}

// High-performance AVX2 implementation for complex_t multiplication
#ifdef __AVX2__
void OpenMPDeconvolutionBackend::complexMultiplicationAVX2(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.data != nullptr, "Input a pointer is null", "OpenMP", "complexMultiplicationAVX2 - input a");
    BACKEND_CHECK(b.data != nullptr, "Input b pointer is null", "OpenMP", "complexMultiplicationAVX2 - input b");
    BACKEND_CHECK(result.data != nullptr, "Result pointer is null", "OpenMP", "complexMultiplicationAVX2 - result");
    
    const int volume = a.size.getVolume();
    const real_t* __restrict__ a_ptr = reinterpret_cast<const real_t*>(a.data);
    const real_t* __restrict__ b_ptr = reinterpret_cast<const real_t*>(b.data);
    real_t* __restrict__ result_ptr = reinterpret_cast<real_t*>(result.data);
    
    const int _end = (volume * 2) & ~7; // Process 4 complex_t numbers (8 real_ts) at a time
    
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for (int i = 0; i < _end; i += 8) {
        // Load 4 complex_t numbers from a and b
        __m256d a_vals = _mm256_load_pd(&a_ptr[i]);     // a0_r, a0_i, a1_r, a1_i
        __m256d a_vals2 = _mm256_load_pd(&a_ptr[i+4]);  // a2_r, a2_i, a3_r, a3_i
        __m256d b_vals = _mm256_load_pd(&b_ptr[i]);     // b0_r, b0_i, b1_r, b1_i
        __m256d b_vals2 = _mm256_load_pd(&b_ptr[i+4]);  // b2_r, b2_i, b3_r, b3_i
        
        // Shuffle to separate real and imaginary parts
        __m256d a_real = _mm256_shuffle_pd(a_vals, a_vals2, 0x0); // a0_r, a1_r, a2_r, a3_r
        __m256d a_imag = _mm256_shuffle_pd(a_vals, a_vals2, 0xF); // a0_i, a1_i, a2_i, a3_i
        __m256d b_real = _mm256_shuffle_pd(b_vals, b_vals2, 0x0); // b0_r, b1_r, b2_r, b3_r
        __m256d b_imag = _mm256_shuffle_pd(b_vals, b_vals2, 0xF); // b0_i, b1_i, b2_i, b3_i
        
        // Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
        __m256d result_real = _mm256_fmsub_pd(a_real, b_real, _mm256_mul_pd(a_imag, b_imag));
        __m256d result_imag = _mm256_fmadd_pd(a_real, b_imag, _mm256_mul_pd(a_imag, b_real));
        
        // Interleave real and imaginary parts back
        __m256d result1 = _mm256_unpacklo_pd(result_real, result_imag);
        __m256d result2 = _mm256_unpackhi_pd(result_real, result_imag);
        
        _mm256_store_pd(&result_ptr[i], result1);
        _mm256_store_pd(&result_ptr[i+4], result2);
    }
    
    // Handle remaining elements
    for (int i = _end / 2; i < volume; ++i) {
        const real_t real_a = a.data[i][0];
        const real_t imag_a = a.data[i][1];
        const real_t real_b = b.data[i][0];
        const real_t imag_b = b.data[i][1];

        result.data[i][0] = real_a * real_b - imag_a * imag_b;
        result.data[i][1] = real_a * imag_b + imag_a * real_b;
    }
}
#endif



void OpenMPBackend::init(const BackendConfig& config){
    set_backend_logger(config.loggingFunction);
    memoryManager.init(config);
    deconvBackend.init(config);
}