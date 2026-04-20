#include "CPUBackend.h"
#include "dolphinbackend/Exceptions.h"
#include "dolphinbackend/IBackend.h"
#include <algorithm>
#include <format>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>

#ifdef __linux__
#include <unistd.h>
#endif

#include "CPUBackendManager.h"

#ifdef _OPENMP
//
// #include <omp.h>
// #define OMP_STRINGIFY(x) #x
// #define OMP_PRAGMA(x) _Pragma(OMP_STRINGIFY(x))
//
// #define OMP(openmp_directive, useOMP, threads) \
//     if(false && threads>1) OMP_PRAGMA(openmp_directive num_threads(threads))
//
// #else

#define OMP(openmp_directive, useOMP, threads)
#else

#define OMP(openmp_directive, useOMP, threads)

#endif



// if more than one backend then these shouldnt be static
FFTWManager CPUDeconvolutionBackend::fftwManager;
MemoryTracking CPUBackendMemoryManager::cpuMemory;


LogCallback g_logger =[](const std::string& message, LogLevel level){
    std::cout << message << std::endl;
};


// CPUBackendMemoryManager implementation
CPUBackendMemoryManager::CPUBackendMemoryManager(CPUBackendConfig config){

    // Initialize memory tracking using getAccess()
    auto access = cpuMemory.getAccess();
    if (access.data.maxMemorySize == 0) {
        access.data.maxMemorySize = getAvailableMemory();
    }
}

CPUBackendMemoryManager::~CPUBackendMemoryManager() {

}


size_t CPUBackendMemoryManager::staticGetAvailableMemory() {
    // For CPU backend, return available system memory
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
}
void CPUBackendMemoryManager::setMemoryLimit(size_t maxMemorySize) {
    auto access = cpuMemory.getAccess();
    access.data.maxMemorySize = maxMemorySize;
}

void CPUBackendMemoryManager::waitForMemory(size_t requiredSize) const {
    auto access = cpuMemory.getAccess();
    if ((access.data.totalUsedMemory + requiredSize) > access.data.maxMemorySize) {

        throw dolphin::backend::MemoryException("Exceeded set memory constraint", "CPU", requiredSize, "Memory Allocation");
        // g_logger(std::format("CPUBackend out of memory, waiting for memory to free up"), LogLevel::ERROR);
    }
    // backend.memory.memoryCondition.wait(lock, [this, requiredSize]() {
    //     return backend.memory.maxMemorySize == 0 || (backend.memory.totalUsedMemory + requiredSize) <= backend.memory.maxMemorySize;
    // });
}

// CPUBackendMemoryManager implementation
bool CPUBackendMemoryManager::isOnDevice(const void* ptr) const {
    // For CPU backend, all valid pointers are "on backend"
    return ptr != nullptr;
}


RealData CPUBackendMemoryManager::allocateMemoryOnDeviceRealFFTInPlace(const CuboidShape& shape) const{
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
RealData CPUBackendMemoryManager::allocateMemoryOnDeviceReal(const CuboidShape& shape) const {
    std::size_t bytes = shape.getVolume() * sizeof(real_t);

    RealData result{ this, nullptr, shape, shape, bytes, 0};
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}


ComplexData CPUBackendMemoryManager::allocateMemoryOnDeviceComplex(const CuboidShape& shape) const{
    CuboidShape complexShape = shape;
    complexShape.width = complexShape.width / 2 + 1;//TODO this is the shape that is needed in the fftw representation of real valued data in complex space
    CuboidShape originalShape = shape;
    ComplexData result{ this, nullptr, complexShape, originalShape, complexShape.getVolume() * sizeof(complex_t), 0};

    // No extra padding: stride equals the complex width
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}
ComplexData CPUBackendMemoryManager::allocateMemoryOnDeviceComplexFull(const CuboidShape& shape) const{
    ComplexData result{ this, nullptr, shape, shape, shape.getVolume() * sizeof(complex_t), 0};
    IBackendMemoryManager::allocateMemoryOnDevice(result);
    return result;
}
DataView<real_t> CPUBackendMemoryManager::reinterpret(ComplexData& data) const{
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

DataView<complex_t> CPUBackendMemoryManager::reinterpret(RealData& data) const{
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

void* CPUBackendMemoryManager::allocateMemoryOnDevice(size_t requested_size) const {
    // Wait for memory if max memory limit is set
    waitForMemory(requested_size);

    void* data = fftwf_malloc(requested_size);
    MEMORY_ALLOC_CHECK(data, requested_size, "CPU", "allocateMemoryOnDevice");

    // Update memory tracking using getAccess()
    auto access = cpuMemory.getAccess();
    access.data.totalUsedMemory += requested_size;
    g_logger(std::format("Allocated {:.2f} MB on device", requested_size / 1e6), LogLevel::INFO);
    return data;

}



 void* CPUBackendMemoryManager::copyDataToDevice(void* src, size_t size, const CuboidShape& shape) const {
    BACKEND_CHECK(src != nullptr, "Source data pointer is null", "CPU", "copyDataToDevice - source data");
    void* result = allocateMemoryOnDevice(size);
    std::memcpy(result, src, size);
    return result;
}

 void* CPUBackendMemoryManager::moveDataFromDevice(void* src, size_t size, const CuboidShape& shape, const IBackendMemoryManager& destBackend) const {
    BACKEND_CHECK(src != nullptr, "Source data pointer is null", "CPU", "moveDataFromDevice - source data");
    if (&destBackend == this) {
        return src;
    }
    else {
        // For cross-backend transfer, use the destination backend's copy method
        // since cpubackend is the "default" it is simple, be careful how this works for other backends though
        return destBackend.copyDataToDevice(src, size, shape);
    }
}




void CPUBackendMemoryManager::memCopy(void* src, void* dest, size_t size, const CuboidShape& shape) const {
    BACKEND_CHECK(src != nullptr, "Source data pointer is null", "CPU", "memCopy - source data");
    BACKEND_CHECK(dest != nullptr, "Destination data pointer is null", "CPU", "memCopy - destination data");
    std::memcpy(dest, src, size);
}

void CPUBackendMemoryManager::freeMemoryOnDevice(void* ptr, size_t size) const{
    BACKEND_CHECK(ptr != nullptr, "Data pointer is null", "CPU", "freeMemoryOnDevice - data pointer");
    fftwf_free(ptr);

    // Update memory tracking using getAccess()
    auto access = cpuMemory.getAccess();
    if (access.data.totalUsedMemory < size) {
        access.data.totalUsedMemory = static_cast<size_t>(0); // this should never happen
    }
    else {
        access.data.totalUsedMemory -= size;
    }

    ptr = nullptr;
    g_logger(std::format("Deallocated {:.2f} MB on device", size / 1e6), LogLevel::INFO);
}



size_t CPUBackendMemoryManager::getAvailableMemory() const {
    try {
        return staticGetAvailableMemory();
    }
    catch (const std::exception& e) {
        g_logger(std::format("Exception in getAvailableMemory: {}", e.what()), LogLevel::ERROR);
        throw; // Re-throw to propagate the exception
    }
}

size_t CPUBackendMemoryManager::getAllocatedMemory() const {
    auto access = cpuMemory.getAccess();
    return access.data.totalUsedMemory;
}


// #####################################################################################################
// CPUDeconvolutionBackend implementation
CPUDeconvolutionBackend::CPUDeconvolutionBackend(CPUBackendConfig config)
    : config(config) {


    #ifdef  _OPENMP
        // omp_set_num_threads(1);
        // omp_set_nested(0);
    #endif
}

CPUDeconvolutionBackend::~CPUDeconvolutionBackend() {

}


void CPUDeconvolutionBackend::initializePlan(const CuboidShape& shape) {
}



void CPUDeconvolutionBackend::forwardFFT(const ComplexData& in, ComplexData& out) const {
    BACKEND_CHECK(in.getData() != nullptr, "Input data pointer is null", "CPU", "forwardFFT - input data");
    BACKEND_CHECK(out.getData() != nullptr, "Output data pointer is null", "CPU", "forwardFFT - output data");

    bool inPlace = in.getData() == out.getData();
    FFTWPlanDescription description(config.ompThreads, PlanDirection::FORWARD, PlanType::COMPLEX, in.getSize(), inPlace);
    fftwManager.executeForwardFFT(description, reinterpret_cast<fftwf_complex*>(in.getData()), reinterpret_cast<fftwf_complex*>(out.getData()));
}

void CPUDeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    BACKEND_CHECK(in.getData() != nullptr, "Input data pointer is null", "CPU", "backwardFFT - input data");
    BACKEND_CHECK(out.getData() != nullptr, "Output data pointer is null", "CPU", "backwardFFT - output data");

    bool inPlace = in.getData() == out.getData();
    FFTWPlanDescription description(config.ompThreads, PlanDirection::BACKWARD, PlanType::COMPLEX, in.getSize(), inPlace);
    fftwManager.executeBackwardFFT(description, reinterpret_cast<fftwf_complex*>(in.getData()), reinterpret_cast<fftwf_complex*>(out.getData()));

    complex_t normFactor{1.0f / out.getSize().getVolume(), 1.0f / out.getSize().getVolume()};//TESTVALUE
    scalarMultiplication(out, normFactor, out); // Add normalization
}

void CPUDeconvolutionBackend::forwardFFT(const RealData& in, ComplexData& out) const {
    BACKEND_CHECK(in.getData() != nullptr, "Input data pointer is null", "CPU", "forwardFFT - input data");
    BACKEND_CHECK(out.getData() != nullptr, "Output data pointer is null", "CPU", "forwardFFT - output data");

    bool inPlace = in.getData() == (real_t*)out.getData();
    FFTWPlanDescription description(config.ompThreads, PlanDirection::FORWARD, PlanType::REAL, in.getSize(), inPlace);
    fftwManager.executeForwardFFTReal(description, reinterpret_cast<real_t*>(in.getData()), reinterpret_cast<fftwf_complex*>(out.getData()));
}

void CPUDeconvolutionBackend::backwardFFT(const ComplexData& in, RealData& out) const {
    BACKEND_CHECK(in.getData() != nullptr, "Input data pointer is null", "CPU", "backwardFFT - input data");
    BACKEND_CHECK(out.getData() != nullptr, "Output data pointer is null", "CPU", "backwardFFT - output data");

    bool inPlace = in.getData() == (complex_t*)out.getData();
    FFTWPlanDescription description(config.ompThreads, PlanDirection::BACKWARD, PlanType::REAL ,out.getSize(), inPlace);
    fftwManager.executeBackwardFFTReal(description, reinterpret_cast<fftwf_complex*>(in.getData()), reinterpret_cast<real_t*>(out.getData()));

    real_t normFactor{1.0f / out.getSize().getVolume()};
    scalarMultiplication(out, normFactor, out); // Add normalization
}

// void CPUDeconvolutionBackend::octantFourierShift(RealData& data) const {
//     int width = data.getSize().width;
//     int height = data.getSize().height;
//     int depth = data.getSize().depth;
//
//     int halfWidth = width / 2;
//     int halfHeight = height / 2;
//     int halfDepth = depth / 2;
//
//
//
//     for (int z = 0; z < depth; ++z) {
//         int newZ = (z + halfDepth) % depth;
//         for (int y = 0; y < height; ++y) {
//             int newY = (y + halfHeight) % height;
//             for (int x = 0; x < width; ++x) {
//                 int newX = (x + halfWidth) % width;
//
//                 int srcIdx = z * height * width + y * width + x;
//                 int dstIdx = newZ * height * width + newY * width + newX;
//
//                 // data[dstIdx] = temp[srcIdx];
//                 std::swap(data[dstIdx], data[srcIdx]);
//             }
//         }
//     }
// }

void CPUDeconvolutionBackend::octantFourierShift(RealData& data) const {
    int width = data.getSize().width;
    int height = data.getSize().height;
    int depth = data.getSize().depth;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;
    // OMP(omp parallel for, useOMP, nThreads)
    // #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = ((z + halfDepth) % depth) * height * width +
                    ((y + halfHeight) % height) * width +
                    ((x + halfWidth) % width);
                if (idx1 != idx2) {
                    std::swap(data[idx1], data[idx2]);
                }
            }
        }
    }
}


void CPUDeconvolutionBackend::octantFourierShift(ComplexData& data) const {
    int width = data.getSize().width;
    int height = data.getSize().height;
    int depth = data.getSize().depth;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;
    // OMP(omp parallel for, useOMP, nThreads)
    // #pragma omp parallel for num_threads(nThreads) collapse(3)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t idx1 = z * height * width + y * width + x;
                size_t idx2 = ((z + halfDepth) % depth) * height * width +
                    ((y + halfHeight) % height) * width +
                    ((x + halfWidth) % width);
                if (idx1 != idx2) {
                    std::swap(data[idx1][0], data[idx2][0]);
                    std::swap(data[idx1][1], data[idx2][1]);
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::inverseQuadrantShift(ComplexData& data) const {
    int width = data.getSize().width;
    int height = data.getSize().height;
    int depth = data.getSize().depth;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < halfHeight; ++y) {
            for (int x = 0; x < halfWidth; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x + halfWidth);

                std::swap(data[idx1][0], data[idx2][0]);
                std::swap(data[idx1][1], data[idx2][1]);
            }
        }
    }

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < halfHeight; ++y) {
            for (int x = halfWidth; x < width; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x - halfWidth);

                std::swap(data[idx1][0], data[idx2][0]);
                std::swap(data[idx1][1], data[idx2][1]);
            }
        }
    }

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = halfHeight; y < height; ++y) {
            for (int x = 0; x < halfWidth; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x + halfWidth);

                std::swap(data[idx1][0], data[idx2][0]);
                std::swap(data[idx1][1], data[idx2][1]);
            }
        }
    }

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = halfHeight; y < height; ++y) {
            for (int x = halfWidth; x < width; ++x) {
                int idx1 = z * height * width + y * width + x;
                int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x - halfWidth);

                std::swap(data[idx1][0], data[idx2][0]);
                std::swap(data[idx1][1], data[idx2][1]);
            }
        }
    }
}

void CPUDeconvolutionBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexMultiplication - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexMultiplication - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexMultiplication - result");


    // OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        real_t real_a = a[i][0];
        real_t imag_a = a[i][1];
        real_t real_b = b[i][0];
        real_t imag_b = b[i][1];

        result[i][0] = real_a * real_b - imag_a * imag_b;
        result[i][1] = real_a * imag_b + imag_a * real_b;
    }
}

void CPUDeconvolutionBackend::multiplication(const RealData& a, const RealData& b, RealData& result) const{

    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexMultiplication - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexMultiplication - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexMultiplication - result");


    // OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        result[i] = a[i] * b[i];
    }
}

void CPUDeconvolutionBackend::division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexDivision - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexDivision - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexDivision - result");

    // OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        real_t denominator = b[i] < epsilon ? epsilon : b[i];
        result[i] = a[i] / denominator;
    }
}


void CPUDeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexDivision - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexDivision - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexDivision - result");


    // OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        real_t real_a = a[i][0];
        real_t imag_a = a[i][1];
        real_t real_b = b[i][0];
        real_t imag_b = b[i][1];

        real_t denominator = real_b * real_b + imag_b * imag_b;

        if (denominator < epsilon) {
            result[i][0] = 0.0;
            result[i][1] = 0.0;
        }
        else {
            result[i][0] = (real_a * real_b + imag_a * imag_b) / denominator;
            result[i][1] = (imag_a * real_b - real_a * imag_b) / denominator;
        }
    }
}


void CPUDeconvolutionBackend::complexAddition(complex_t** data, ComplexData& sum, int nImages, int imageVolume) const {
    BACKEND_CHECK(sum.getData() != nullptr, "Input b pointer is null", "CPU", "complexAddition - input b");

    int imageSize = sum.getSize().getVolume();
    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int imageindex = 0; imageindex < nImages; ++imageindex) {

        complex_t* a = data[imageindex];
        for (int i = 0; i < imageSize; ++i) {
            // Use atomic to prevent race conditions when multiple threads write to sum[i]
            #pragma omp atomic
            sum[i][0] += a[i][0];
            #pragma omp atomic
            sum[i][1] += a[i][1];
        }
    }
}

void CPUDeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexAddition - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexAddition - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexAddition - result");

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        result[i][0] = a[i][0] + b[i][0];
        result[i][1] = a[i][1] + b[i][1];
    }
}

void CPUDeconvolutionBackend::sumToOne(real_t** data, int nImages, int imageVolume) const {

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < imageVolume; ++i) {
        real_t sum{ 0 };

        for (int imageindex = 0; imageindex < nImages; ++imageindex) {
            sum += data[imageindex][i];
        }

        for (int imageindex = 0; imageindex < nImages; ++imageindex) {
            if (sum == 0.0f) data[imageindex][i] = 0.0f;
            data[imageindex][i] /= sum;
        }
    }
}
void CPUDeconvolutionBackend::scalarMultiplication(const RealData& a, real_t scalar, RealData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "scalarMultiplication - input a");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "scalarMultiplication - result");

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        result[i] = a[i] * scalar;
    }
}

void CPUDeconvolutionBackend::scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "scalarMultiplication - input a");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "scalarMultiplication - result");

    real_t rscalar = scalar[0];
    real_t iscalar = scalar[1];
    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        result[i][0] = a[i][0] * rscalar;
        result[i][1] = a[i][1] * iscalar;
    }
}

void CPUDeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexMultiplicationWithConjugate - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexMultiplicationWithConjugate - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexMultiplicationWithConjugate - result");


    // OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        real_t real_a = a[i][0];
        real_t imag_a = a[i][1];
        real_t real_b = b[i][0];
        real_t imag_b = -b[i][1];  // Conjugate

        result[i][0] = real_a * real_b - imag_a * imag_b;
        result[i][1] = real_a * imag_b + imag_a * real_b;
    }
}

void CPUDeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexDivisionStabilized - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexDivisionStabilized - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexDivisionStabilized - result");


    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < a.getSize().getVolume(); ++i) {
        real_t real_a = a[i][0];
        real_t imag_a = a[i][1];
        real_t real_b = b[i][0];
        real_t imag_b = b[i][1];

        real_t mag = std::max(epsilon, real_b * real_b + imag_b * imag_b);

        result[i][0] = (real_a * real_b + imag_a * imag_b) / mag;
        result[i][1] = (imag_a * real_b - real_a * imag_b) / mag;
    }
}

// Specialized Functions
void CPUDeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
    int width = psf.getSize().width;
    int height = psf.getSize().height;
    int depth = psf.getSize().depth;

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        float wz = 2 * M_PI * z / depth;
        for (int y = 0; y < height; ++y) {
            float wy = 2 * M_PI * y / height;
            for (int x = 0; x < width; ++x) {
                float wx = 2 * M_PI * x / width;
                float laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

                int index = (z * height + y) * width + x;

                laplacian[index][0] = psf[index][0] * laplacian_value;
                laplacian[index][1] = psf[index][1] * laplacian_value;
            }
        }
    }
}

// void CPUDeconvolutionBackend::normalizeImage(ComplexData& resultImage, real_t epsilon) const {
//     real_t max_val = 0.0, max_val2 = 0.0;
//     OMP(omp parallel for, config.useOMP, config.ompThreads)
//     for (int j = 0; j < resultImage.getSize().getVolume(); j++) {
//         max_val = std::max(max_val, resultImage[j][0]);
//         max_val2 = std::max(max_val2, resultImage[j][1]);
//     }
//     OMP(omp parallel for, config.useOMP, config.ompThreads)
//     for (int j = 0; j < resultImage.getSize().getVolume(); j++) {
//         resultImage[j][0] /= (max_val + epsilon);
//         resultImage[j][1] /= (max_val2 + epsilon);
//     }
// }
//
// void CPUDeconvolutionBackend::rescaledInverse(ComplexData& data, real_t cubeVolume) const {
//     OMP(omp parallel for, config.useOMP, config.ompThreads)
//     for (int i = 0; i < data.getSize().getVolume(); ++i) {
//         data[i][0] /= cubeVolume;
//         data[i][1] /= cubeVolume;
//     }
// }

// Debug functions
void CPUDeconvolutionBackend::hasNAN(const ComplexData& data) const {
    int nanCount = 0, infCount = 0;
    real_t minReal = std::numeric_limits<real_t>::max();
    real_t maxReal = std::numeric_limits<real_t>::lowest();
    real_t minImag = std::numeric_limits<real_t>::max();
    real_t maxImag = std::numeric_limits<real_t>::lowest();

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int i = 0; i < data.getSize().getVolume(); i++) {
        real_t real = data[i][0];
        real_t imag = data[i][1];

        // Check for NaN
        if (std::isnan(real) || std::isnan(imag)) {
            nanCount++;
            if (nanCount <= 10) { // Only print first 10
                g_logger(std::format("NaN at index {}: ({}, {})", i, real, imag), LogLevel::INFO);
            }
        }

        // Check for infinity
        if (std::isinf(real) || std::isinf(imag)) {
            infCount++;
            if (infCount <= 10) {
                g_logger(std::format("Inf at index {}: ({}, {})", i, real, imag), LogLevel::INFO);
            }
        }

        // Track min/max for valid values
        if (std::isfinite(real)) {
            minReal = std::min(minReal, real);
            maxReal = std::max(maxReal, real);
        }
        if (std::isfinite(imag)) {
            minImag = std::min(minImag, imag);
            maxImag = std::max(maxImag, imag);
        }
    }

    g_logger(std::format("Data stats - NaN: {}, Inf: {}", nanCount, infCount), LogLevel::INFO);
    g_logger(std::format("Real range: [{}, {}]", minReal, maxReal), LogLevel::INFO);
    g_logger(std::format("Imag range: [{}, {}]", minImag, maxImag), LogLevel::INFO);
}


// Gradient and TV Functions - Updated to match OpenMPBackend pattern
void CPUDeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    int width = image.getSize().width;
    int height = image.getSize().height;
    int depth = image.getSize().depth;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;

                if (x < width - 1) {
                    int nextIndex = index + 1;
                    gradX[index][0] = image[index][0] - image[nextIndex][0];
                    gradX[index][1] = image[index][1] - image[nextIndex][1];
                } else {
                    // Boundary condition: last column
                    gradX[index][0] = 0.0;
                    gradX[index][1] = 0.0;
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    int width = image.getSize().width;
    int height = image.getSize().height;
    int depth = image.getSize().depth;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;

                if (y < height - 1) {
                    int nextIndex = index + width;
                    gradY[index][0] = image[index][0] - image[nextIndex][0];
                    gradY[index][1] = image[index][1] - image[nextIndex][1];
                } else {
                    // Boundary condition: last row
                    gradY[index][0] = 0.0;
                    gradY[index][1] = 0.0;
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    int width = image.getSize().width;
    int height = image.getSize().height;
    int depth = image.getSize().depth;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;

                if (z < depth - 1) {
                    int nextIndex = index + height * width;
                    gradZ[index][0] = image[index][0] - image[nextIndex][0];
                    gradZ[index][1] = image[index][1] - image[nextIndex][1];
                } else {
                    // Boundary condition: last depth layer
                    gradZ[index][0] = 0.0;
                    gradZ[index][1] = 0.0;
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
    int nxy = gx.getSize().width * gx.getSize().height;

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int z = 0; z < gx.getSize().depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            real_t dx = gx[index][0];
            real_t dy = gy[index][0];
            real_t dz = gz[index][0];

            tv[index][0] = static_cast<real_t>(1.0 / (1.0 - ((dx + dy + dz) * lambda)));
            tv[index][1] = 0.0;
        }
    }
}



void CPUDeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const {
    int nxy = gradX.getSize().width * gradX.getSize().height;

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int z = 0; z < gradX.getSize().depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            real_t norm = std::sqrt(
                gradX[index][0] * gradX[index][0] + gradX[index][1] * gradX[index][1] +
                gradY[index][0] * gradY[index][0] + gradY[index][1] * gradY[index][1] +
                gradZ[index][0] * gradZ[index][0] + gradZ[index][1] * gradZ[index][1]
            );

            norm = std::max(norm, epsilon);

            gradX[index][0] /= norm;
            gradX[index][1] /= norm;
            gradY[index][0] /= norm;
            gradY[index][1] /= norm;
            gradZ[index][0] /= norm;
            gradZ[index][1] /= norm;
        }
    }
}

// Gradient and TV Functions for real-valued data
void CPUDeconvolutionBackend::gradientX(const RealData& image, RealData& gradX) const {
    int width = image.getSize().width;
    int height = image.getSize().height;
    int depth = image.getSize().depth;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;

                if (x < width - 1) {
                    int nextIndex = index + 1;
                    gradX[index] = image[index] - image[nextIndex];
                } else {
                    // Boundary condition: last column
                    gradX[index] = 0.0;
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::gradientY(const RealData& image, RealData& gradY) const {
    int width = image.getSize().width;
    int height = image.getSize().height;
    int depth = image.getSize().depth;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;

                if (y < height - 1) {
                    int nextIndex = index + width;
                    gradY[index] = image[index] - image[nextIndex];
                } else {
                    // Boundary condition: last row
                    gradY[index] = 0.0;
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::gradientZ(const RealData& image, RealData& gradZ) const {
    int width = image.getSize().width;
    int height = image.getSize().height;
    int depth = image.getSize().depth;

    OMP(omp parallel for collapse(3), config.useOMP, config.ompThreads)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;

                if (z < depth - 1) {
                    int nextIndex = index + height * width;
                    gradZ[index] = image[index] - image[nextIndex];
                } else {
                    // Boundary condition: last depth layer
                    gradZ[index] = 0.0;
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::computeTV(real_t lambda, const RealData& gx, const RealData& gy, const RealData& gz, RealData& tv) const {
    int nxy = gx.getSize().width * gx.getSize().height;

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int z = 0; z < gx.getSize().depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            real_t dx = gx[index];
            real_t dy = gy[index];
            real_t dz = gz[index];

            tv[index] = static_cast<real_t>(1.0 / (1.0 - ((dx + dy + dz) * lambda)));
        }
    }
}

void CPUDeconvolutionBackend::normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t epsilon) const {
    int nxy = gradX.getSize().width * gradX.getSize().height;

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int z = 0; z < gradX.getSize().depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            real_t norm = std::sqrt(
                gradX[index] * gradX[index] +
                gradY[index] * gradY[index] +
                gradZ[index] * gradZ[index]
            );

            norm = std::max(norm, epsilon);

            gradX[index] /= norm;
            gradY[index] /= norm;
            gradZ[index] /= norm;
        }
    }
}


// IBackend& CPUBackend::clone() {
//     return backendManager.clone(*this);
// }

// IBackend& CPUBackend::cloneSharedMemory() {
//     return backendManager.cloneSharedMemory(*this);
// }

// void CPUBackend::setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads) {
//     // workerThreads = static_cast<size_t>(2*totalThreads/3);
//     // config.ompThreads = static_cast<int>(workerThreads);

//     ioThreads = ioThreads == 0 ? totalThreads : ioThreads;
//     config.ompThreads = workerThreads == 0 ? static_cast<size_t>(2*totalThreads/3) : workerThreads;
//     // workerThreads = workerThreads == 0 ? 1 : workerThreads;
//     workerThreads = 1;
//     deconvDevice.init(config);
// }
