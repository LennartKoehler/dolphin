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



// Strided iteration helpers — iterate over valid elements of 3D strided buffers
// row-by-row, skipping FFTW padding at the end of each row.
// Each buffer can have its own stride layout.
//
// Convention: input parameters are const references; the result parameter is a
// non-const reference. Use stridedIterationMutate when all parameters are modified.
//
// Usage examples:
//   // 2-input (a, result):
//   stridedIteration(a, result, [&](auto* rowA, auto* rowR, int w) {
//       for (int x = 0; x < w; ++x) rowR[x] = rowA[x] * scalar;
//   });
//   // 3-input (a, b, result):
//   stridedIteration(a, b, result, [&](auto* rowA, auto* rowB, auto* rowR, int w) {
//       for (int x = 0; x < w; ++x) rowR[x] = rowA[x] * rowB[x];
//   });

// Stride info extracted from a ManagedData/DataView object
struct StrideInfo {
    int width;
    int height;
    int depth;
    size_t stride;       // elements per row  (width + padding)
    size_t sliceStride;  // elements per slice (stride * height)
};

template<typename DataRef>
StrideInfo getStrideInfo(const DataRef& d) {
    StrideInfo info;
    info.width       = d.getSize().width;
    info.height      = d.getSize().height;
    info.depth       = d.getSize().depth;
    info.stride      = static_cast<size_t>(info.width) + d.getPadding();
    info.sliceStride = info.stride * info.height;
    return info;
}

// --- 1-input ---
template<typename D1, typename Func>
void stridedIteration(const D1& d1, Func&& func) {
    auto si1 = getStrideInfo(d1);
    auto* p1 = d1.getData();
    for (int z = 0; z < si1.depth; ++z)
        for (int y = 0; y < si1.height; ++y)
            func(p1 + z * si1.sliceStride + y * si1.stride, si1.width);
}

// --- 2-input (1 input + result) ---
template<typename D1, typename DR, typename Func>
void stridedIteration(const D1& d1, DR& result, Func&& func) {
    auto si1 = getStrideInfo(d1);
    auto siR = getStrideInfo(result);
    const auto* p1 = d1.getData();
    auto* pR = result.getData();
    for (int z = 0; z < si1.depth; ++z)
        for (int y = 0; y < si1.height; ++y) {
            auto off1 = z * si1.sliceStride + y * si1.stride;
            auto offR = z * siR.sliceStride + y * siR.stride;
            func(p1 + off1, pR + offR, si1.width);
        }
}

// --- 3-input (2 inputs + result) ---
template<typename D1, typename D2, typename DR, typename Func>
void stridedIteration(const D1& d1, const D2& d2, DR& result, Func&& func) {
    auto si1 = getStrideInfo(d1);
    auto si2 = getStrideInfo(d2);
    auto siR = getStrideInfo(result);
    const auto* p1 = d1.getData();
    const auto* p2 = d2.getData();
    auto* pR = result.getData();
    for (int z = 0; z < si1.depth; ++z)
        for (int y = 0; y < si1.height; ++y) {
            auto off1 = z * si1.sliceStride + y * si1.stride;
            auto off2 = z * si2.sliceStride + y * si2.stride;
            auto offR = z * siR.sliceStride + y * siR.stride;
            func(p1 + off1, p2 + off2, pR + offR, si1.width);
        }
}

// --- 4-input (3 inputs + result) ---
template<typename D1, typename D2, typename D3, typename DR, typename Func>
void stridedIteration(const D1& d1, const D2& d2, const D3& d3, DR& result, Func&& func) {
    auto si1 = getStrideInfo(d1);
    auto si2 = getStrideInfo(d2);
    auto si3 = getStrideInfo(d3);
    auto siR = getStrideInfo(result);
    const auto* p1 = d1.getData();
    const auto* p2 = d2.getData();
    const auto* p3 = d3.getData();
    auto* pR = result.getData();
    for (int z = 0; z < si1.depth; ++z)
        for (int y = 0; y < si1.height; ++y) {
            auto off1 = z * si1.sliceStride + y * si1.stride;
            auto off2 = z * si2.sliceStride + y * si2.stride;
            auto off3 = z * si3.sliceStride + y * si3.stride;
            auto offR = z * siR.sliceStride + y * siR.stride;
            func(p1 + off1, p2 + off2, p3 + off3, pR + offR, si1.width);
        }
}

// --- 3-mutable (all three modified, e.g. normalizeTV) ---
template<typename D1, typename D2, typename D3, typename Func>
void stridedIterationMutate(D1& d1, D2& d2, D3& d3, Func&& func) {
    auto si1 = getStrideInfo(d1);
    auto si2 = getStrideInfo(d2);
    auto si3 = getStrideInfo(d3);
    auto* p1 = d1.getData();
    auto* p2 = d2.getData();
    auto* p3 = d3.getData();
    for (int z = 0; z < si1.depth; ++z)
        for (int y = 0; y < si1.height; ++y) {
            auto off1 = z * si1.sliceStride + y * si1.stride;
            auto off2 = z * si2.sliceStride + y * si2.stride;
            auto off3 = z * si3.sliceStride + y * si3.stride;
            func(p1 + off1, p2 + off2, p3 + off3, si1.width);
        }
}



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

    stridedIteration(a, b, result, [](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            real_t ra = rowA[x][0], ia = rowA[x][1];
            real_t rb = rowB[x][0], ib = rowB[x][1];
            rowR[x][0] = ra * rb - ia * ib;
            rowR[x][1] = ra * ib + ia * rb;
        }
    });
}

void CPUDeconvolutionBackend::multiplication(const RealData& a, const RealData& b, RealData& result) const{
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "multiplication - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "multiplication - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "multiplication - result");

    stridedIteration(a, b, result, [](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x)
            rowR[x] = rowA[x] * rowB[x];
    });
}

void CPUDeconvolutionBackend::division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "division - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "division - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "division - result");

    stridedIteration(a, b, result, [epsilon](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            real_t denom = rowB[x] < epsilon ? epsilon : rowB[x];
            rowR[x] = rowA[x] / denom;
        }
    });
}


void CPUDeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexDivision - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexDivision - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexDivision - result");

    stridedIteration(a, b, result, [epsilon](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            real_t ra = rowA[x][0], ia = rowA[x][1];
            real_t rb = rowB[x][0], ib = rowB[x][1];
            real_t denom = rb * rb + ib * ib;
            if (denom < epsilon) {
                rowR[x][0] = 0.0;
                rowR[x][1] = 0.0;
            } else {
                rowR[x][0] = (ra * rb + ia * ib) / denom;
                rowR[x][1] = (ia * rb - ra * ib) / denom;
            }
        }
    });
}


void CPUDeconvolutionBackend::complexAddition(complex_t** data, ComplexData& sum, int nImages, int imageVolume) const {
    BACKEND_CHECK(sum.getData() != nullptr, "Input b pointer is null", "CPU", "complexAddition - input b");

    auto si = getStrideInfo(sum);
    complex_t* ptrSum = sum.getData();

    OMP(omp parallel for, config.useOMP, config.ompThreads)
    for (int imageindex = 0; imageindex < nImages; ++imageindex) {
        complex_t* a = data[imageindex];
        for (int z = 0; z < si.depth; ++z) {
            for (int y = 0; y < si.height; ++y) {
                size_t off = z * si.sliceStride + y * si.stride;
                for (int x = 0; x < si.width; ++x) {
                    #pragma omp atomic
                    ptrSum[off + x][0] += a[off + x][0];
                    #pragma omp atomic
                    ptrSum[off + x][1] += a[off + x][1];
                }
            }
        }
    }
}

void CPUDeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexAddition - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexAddition - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexAddition - result");

    stridedIteration(a, b, result, [](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            rowR[x][0] = rowA[x][0] + rowB[x][0];
            rowR[x][1] = rowA[x][1] + rowB[x][1];
        }
    });
}

void CPUDeconvolutionBackend::sumToOne(real_t** data, int nImages, int imageVolume) const {
    // NOTE: Uses raw real_t** with flat indexing. External pointers are assumed
    // to share the same stride layout but we don't have CuboidShape for them.
    // If stride-aware access is needed, the API should be updated to pass shapes.
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

    stridedIteration(a, result, [scalar](auto* rowA, auto* rowR, int w) {
        for (int x = 0; x < w; ++x)
            rowR[x] = rowA[x] * scalar;
    });
}

void CPUDeconvolutionBackend::scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "scalarMultiplication - input a");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "scalarMultiplication - result");

    const real_t rscalar = scalar[0];
    const real_t iscalar = scalar[1];
    stridedIteration(a, result, [rscalar, iscalar](auto* rowA, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            rowR[x][0] = rowA[x][0] * rscalar;
            rowR[x][1] = rowA[x][1] * iscalar;
        }
    });
}

void CPUDeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexMultiplicationWithConjugate - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexMultiplicationWithConjugate - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexMultiplicationWithConjugate - result");

    stridedIteration(a, b, result, [](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            real_t ra = rowA[x][0], ia = rowA[x][1];
            real_t rb = rowB[x][0], ib = -rowB[x][1];  // Conjugate
            rowR[x][0] = ra * rb - ia * ib;
            rowR[x][1] = ra * ib + ia * rb;
        }
    });
}

void CPUDeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
    BACKEND_CHECK(a.getData() != nullptr, "Input a pointer is null", "CPU", "complexDivisionStabilized - input a");
    BACKEND_CHECK(b.getData() != nullptr, "Input b pointer is null", "CPU", "complexDivisionStabilized - input b");
    BACKEND_CHECK(result.getData() != nullptr, "Result pointer is null", "CPU", "complexDivisionStabilized - result");

    stridedIteration(a, b, result, [epsilon](auto* rowA, auto* rowB, auto* rowR, int w) {
        for (int x = 0; x < w; ++x) {
            real_t ra = rowA[x][0], ia = rowA[x][1];
            real_t rb = rowB[x][0], ib = rowB[x][1];
            real_t mag = std::max(epsilon, rb * rb + ib * ib);
            rowR[x][0] = (ra * rb + ia * ib) / mag;
            rowR[x][1] = (ia * rb - ra * ib) / mag;
        }
    });
}

// // Specialized Functions
// void CPUDeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
//     auto siPsf = getStrideInfo(psf);
//     auto siLap = getStrideInfo(laplacian);
//     const complex_t* ptrPsf = psf.getData();
//     complex_t*       ptrLap = laplacian.getData();
//
//     OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
//     for (int z = 0; z < siPsf.depth; ++z) {
//         float wz = 2 * M_PI * z / siPsf.depth;
//         for (int y = 0; y < siPsf.height; ++y) {
//             float wy = 2 * M_PI * y / siPsf.height;
//             auto offPsf = z * siPsf.sliceStride + y * siPsf.stride;
//             auto offLap = z * siLap.sliceStride + y * siLap.stride;
//             for (int x = 0; x < siPsf.width; ++x) {
//                 float wx = 2 * M_PI * x / siPsf.width;
//                 float lap_val = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);
//                 ptrLap[offLap + x][0] = ptrPsf[offPsf + x][0] * lap_val;
//                 ptrLap[offLap + x][1] = ptrPsf[offPsf + x][1] * lap_val;
//             }
//         }
//     }
// }
//
// // void CPUDeconvolutionBackend::normalizeImage(ComplexData& resultImage, real_t epsilon) const {
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

    auto si = getStrideInfo(data);
    const complex_t* ptr = data.getData();

    for (int z = 0; z < si.depth; ++z) {
        for (int y = 0; y < si.height; ++y) {
            size_t off = z * si.sliceStride + y * si.stride;
            for (int x = 0; x < si.width; ++x) {
                real_t rv = ptr[off + x][0];
                real_t iv = ptr[off + x][1];

                if (std::isnan(rv) || std::isnan(iv)) {
                    nanCount++;
                    if (nanCount <= 10)
                        g_logger(std::format("NaN at ({},{},{}): ({}, {})", x, y, z, rv, iv), LogLevel::INFO);
                }
                if (std::isinf(rv) || std::isinf(iv)) {
                    infCount++;
                    if (infCount <= 10)
                        g_logger(std::format("Inf at ({},{},{}): ({}, {})", x, y, z, rv, iv), LogLevel::INFO);
                }
                if (std::isfinite(rv)) {
                    minReal = std::min(minReal, rv);
                    maxReal = std::max(maxReal, rv);
                }
                if (std::isfinite(iv)) {
                    minImag = std::min(minImag, iv);
                    maxImag = std::max(maxImag, iv);
                }
            }
        }
    }

    g_logger(std::format("Data stats - NaN: {}, Inf: {}", nanCount, infCount), LogLevel::INFO);
    g_logger(std::format("Real range: [{}, {}]", minReal, maxReal), LogLevel::INFO);
    g_logger(std::format("Imag range: [{}, {}]", minImag, maxImag), LogLevel::INFO);
}


// Gradient and TV Functions - Updated to match OpenMPBackend pattern
void CPUDeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    auto siImg = getStrideInfo(image);
    auto siGrd = getStrideInfo(gradX);
    const complex_t* ptrImg = image.getData();
    complex_t*       ptrGrd = gradX.getData();

    OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
    for (int z = 0; z < siImg.depth; ++z) {
        for (int y = 0; y < siImg.height; ++y) {
            auto offImg = z * siImg.sliceStride + y * siImg.stride;
            auto offGrd = z * siGrd.sliceStride + y * siGrd.stride;
            for (int x = 0; x < siImg.width - 1; ++x) {
                ptrGrd[offGrd + x][0] = ptrImg[offImg + x][0] - ptrImg[offImg + x + 1][0];
                ptrGrd[offGrd + x][1] = ptrImg[offImg + x][1] - ptrImg[offImg + x + 1][1];
            }
            // Boundary: last column
            ptrGrd[offGrd + siImg.width - 1][0] = 0.0;
            ptrGrd[offGrd + siImg.width - 1][1] = 0.0;
        }
    }
}

void CPUDeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    auto siImg = getStrideInfo(image);
    auto siGrd = getStrideInfo(gradY);
    const complex_t* ptrImg = image.getData();
    complex_t*       ptrGrd = gradY.getData();

    OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
    for (int z = 0; z < siImg.depth; ++z) {
        for (int y = 0; y < siImg.height - 1; ++y) {
            auto offImg = z * siImg.sliceStride + y * siImg.stride;
            auto offGrd = z * siGrd.sliceStride + y * siGrd.stride;
            for (int x = 0; x < siImg.width; ++x) {
                ptrGrd[offGrd + x][0] = ptrImg[offImg + x][0] - ptrImg[offImg + siImg.stride + x][0];
                ptrGrd[offGrd + x][1] = ptrImg[offImg + x][1] - ptrImg[offImg + siImg.stride + x][1];
            }
        }
        // Boundary: last row
        auto offImg = z * siImg.sliceStride + (siImg.height - 1) * siImg.stride;
        auto offGrd = z * siGrd.sliceStride + (siImg.height - 1) * siGrd.stride;
        for (int x = 0; x < siImg.width; ++x) {
            ptrGrd[offGrd + x][0] = 0.0;
            ptrGrd[offGrd + x][1] = 0.0;
        }
    }
}

void CPUDeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    auto siImg = getStrideInfo(image);
    auto siGrd = getStrideInfo(gradZ);
    const complex_t* ptrImg = image.getData();
    complex_t*       ptrGrd = gradZ.getData();

    OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
    for (int z = 0; z < siImg.depth - 1; ++z) {
        for (int y = 0; y < siImg.height; ++y) {
            auto offImg = z * siImg.sliceStride + y * siImg.stride;
            auto offGrd = z * siGrd.sliceStride + y * siGrd.stride;
            for (int x = 0; x < siImg.width; ++x) {
                ptrGrd[offGrd + x][0] = ptrImg[offImg + x][0] - ptrImg[offImg + siImg.sliceStride + x][0];
                ptrGrd[offGrd + x][1] = ptrImg[offImg + x][1] - ptrImg[offImg + siImg.sliceStride + x][1];
            }
        }
    }
    // Boundary: last slice
    for (int y = 0; y < siImg.height; ++y) {
        auto offImg = (siImg.depth - 1) * siImg.sliceStride + y * siImg.stride;
        auto offGrd = (siImg.depth - 1) * siGrd.sliceStride + y * siGrd.stride;
        for (int x = 0; x < siImg.width; ++x) {
            ptrGrd[offGrd + x][0] = 0.0;
            ptrGrd[offGrd + x][1] = 0.0;
        }
    }
}

void CPUDeconvolutionBackend::computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
    stridedIteration(gx, gy, gz, tv, [lambda](auto* rowGx, auto* rowGy, auto* rowGz, auto* rowTv, int w) {
        for (int x = 0; x < w; ++x) {
            real_t dx = rowGx[x][0];
            real_t dy = rowGy[x][0];
            real_t dz = rowGz[x][0];
            rowTv[x][0] = static_cast<real_t>(1.0 / (1.0 - ((dx + dy + dz) * lambda)));
            rowTv[x][1] = 0.0;
        }
    });
}



void CPUDeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const {
    stridedIterationMutate(gradX, gradY, gradZ, [epsilon](auto* rowGx, auto* rowGy, auto* rowGz, int w) {
        for (int x = 0; x < w; ++x) {
            real_t norm = std::sqrt(
                rowGx[x][0] * rowGx[x][0] + rowGx[x][1] * rowGx[x][1] +
                rowGy[x][0] * rowGy[x][0] + rowGy[x][1] * rowGy[x][1] +
                rowGz[x][0] * rowGz[x][0] + rowGz[x][1] * rowGz[x][1]
            );
            norm = std::max(norm, epsilon);
            rowGx[x][0] /= norm; rowGx[x][1] /= norm;
            rowGy[x][0] /= norm; rowGy[x][1] /= norm;
            rowGz[x][0] /= norm; rowGz[x][1] /= norm;
        }
    });
}

// Gradient and TV Functions for real-valued data
void CPUDeconvolutionBackend::gradientX(const RealData& image, RealData& gradX) const {
    auto siImg = getStrideInfo(image);
    auto siGrd = getStrideInfo(gradX);
    const real_t* ptrImg = image.getData();
    real_t*       ptrGrd = gradX.getData();

    OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
    for (int z = 0; z < siImg.depth; ++z) {
        for (int y = 0; y < siImg.height; ++y) {
            auto offImg = z * siImg.sliceStride + y * siImg.stride;
            auto offGrd = z * siGrd.sliceStride + y * siGrd.stride;
            for (int x = 0; x < siImg.width - 1; ++x)
                ptrGrd[offGrd + x] = ptrImg[offImg + x] - ptrImg[offImg + x + 1];
            ptrGrd[offGrd + siImg.width - 1] = 0.0;
        }
    }
}

void CPUDeconvolutionBackend::gradientY(const RealData& image, RealData& gradY) const {
    auto siImg = getStrideInfo(image);
    auto siGrd = getStrideInfo(gradY);
    const real_t* ptrImg = image.getData();
    real_t*       ptrGrd = gradY.getData();

    OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
    for (int z = 0; z < siImg.depth; ++z) {
        for (int y = 0; y < siImg.height - 1; ++y) {
            auto offImg = z * siImg.sliceStride + y * siImg.stride;
            auto offGrd = z * siGrd.sliceStride + y * siGrd.stride;
            for (int x = 0; x < siImg.width; ++x)
                ptrGrd[offGrd + x] = ptrImg[offImg + x] - ptrImg[offImg + siImg.stride + x];
        }
        // Boundary: last row
        auto offGrd = z * siGrd.sliceStride + (siImg.height - 1) * siGrd.stride;
        for (int x = 0; x < siImg.width; ++x)
            ptrGrd[offGrd + x] = 0.0;
    }
}

void CPUDeconvolutionBackend::gradientZ(const RealData& image, RealData& gradZ) const {
    auto siImg = getStrideInfo(image);
    auto siGrd = getStrideInfo(gradZ);
    const real_t* ptrImg = image.getData();
    real_t*       ptrGrd = gradZ.getData();

    OMP(omp parallel for collapse(2), config.useOMP, config.ompThreads)
    for (int z = 0; z < siImg.depth - 1; ++z) {
        for (int y = 0; y < siImg.height; ++y) {
            auto offImg = z * siImg.sliceStride + y * siImg.stride;
            auto offGrd = z * siGrd.sliceStride + y * siGrd.stride;
            for (int x = 0; x < siImg.width; ++x)
                ptrGrd[offGrd + x] = ptrImg[offImg + x] - ptrImg[offImg + siImg.sliceStride + x];
        }
    }
    // Boundary: last slice
    for (int y = 0; y < siImg.height; ++y) {
        auto offGrd = (siImg.depth - 1) * siGrd.sliceStride + y * siGrd.stride;
        for (int x = 0; x < siImg.width; ++x)
            ptrGrd[offGrd + x] = 0.0;
    }
}

void CPUDeconvolutionBackend::computeTV(real_t lambda, const RealData& gx, const RealData& gy, const RealData& gz, RealData& tv) const {
    stridedIteration(gx, gy, gz, tv, [lambda](auto* rowGx, auto* rowGy, auto* rowGz, auto* rowTv, int w) {
        for (int x = 0; x < w; ++x) {
            real_t dx = rowGx[x];
            real_t dy = rowGy[x];
            real_t dz = rowGz[x];
            rowTv[x] = static_cast<real_t>(1.0 / (1.0 - ((dx + dy + dz) * lambda)));
        }
    });
}

void CPUDeconvolutionBackend::normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t epsilon) const {
    stridedIterationMutate(gradX, gradY, gradZ, [epsilon](auto* rowGx, auto* rowGy, auto* rowGz, int w) {
        for (int x = 0; x < w; ++x) {
            real_t norm = std::sqrt(
                rowGx[x] * rowGx[x] +
                rowGy[x] * rowGy[x] +
                rowGz[x] * rowGz[x]
            );
            norm = std::max(norm, epsilon);
            rowGx[x] /= norm;
            rowGy[x] /= norm;
            rowGz[x] /= norm;
        }
    });
}

