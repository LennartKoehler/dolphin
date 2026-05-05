#include <iostream>
#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <thread>
#include <atomic>
#include <limits>

#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"

// ============================================================================
// Simple test helpers
// ============================================================================

static int g_testsPassed = 0;
static int g_testsFailed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  FAIL: " << msg << " (at line " << __LINE__ << ")" << std::endl; \
            g_testsFailed++; \
            return; \
        } \
    } while(0)

#define TEST_ASSERT_NO_THROW(expr, msg) \
    do { \
        try { expr; } \
        catch (const std::exception& e) { \
            std::cerr << "  FAIL: " << msg << " - threw: " << e.what() << std::endl; \
            g_testsFailed++; \
            return; \
        } \
    } while(0)

#define RUN_TEST(func) \
    do { \
        std::cout << "Running " << #func << "..." << std::endl; \
        int failsBefore = g_testsFailed; \
        func(); \
        if (g_testsFailed == failsBefore) { \
            std::cout << "  PASSED" << std::endl; \
            g_testsPassed++; \
        } \
    } while(0)

static bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool approxEqualComplex(const complex_t& a, float real, float imag, float eps = 1e-4f) {
    return approxEqual(a[0], real, eps) && approxEqual(a[1], imag, eps);
}

// ============================================================================
// Shared manager
// ============================================================================

static CPUBackendManager& getManager() {
    static CPUBackendManager manager;
    static bool initialized = false;
    if (!initialized) {
        manager.init([](const std::string& msg, LogLevel level) {
            if (level >= LogLevel::ERROR) {
                std::cerr << "[CPU] " << msg << std::endl;
            }
        });
        initialized = true;
    }
    return manager;
}

// ============================================================================
// Test: CPUBackendManager initialization
// ============================================================================

void testManagerInit() {
    CPUBackendManager& mgr = getManager();
    TEST_ASSERT(mgr.getNumberDevices() == 1, "CPU backend should report 1 device");
}

// ============================================================================
// Test: Backend creation and basic queries
// ============================================================================

void testBackendCreation() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;
    config.backendName = "test";

    IBackend& backend = mgr.getBackend(config);

    TEST_ASSERT(backend.getDeviceString() == "cpu", "Device string should be 'cpu'");
    TEST_ASSERT(backend.hasMemoryManager(), "Backend should have a memory manager");
    TEST_ASSERT(backend.ownsDeconvolutionBackend(), "Backend should own its deconv backend");
    TEST_ASSERT(backend.ownsMemoryManager(), "Backend should own its memory manager");
    TEST_ASSERT(backend.getMemoryManagerPtr() != nullptr, "Memory manager pointer should not be null");

    const IDeconvolutionBackend& deconv = backend.getDeconvManager();
    TEST_ASSERT(deconv.getDeviceString() == "cpu", "Deconv device string should be 'cpu'");

    const IBackendMemoryManager& mem = backend.getMemoryManager();
    TEST_ASSERT(mem.getDeviceString() == "cpu", "Memory manager device string should be 'cpu'");
}

// ============================================================================
// Test: Memory allocation and deallocation
// ============================================================================

void testMemoryAllocation() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);

    // Allocate real data
    RealData realData = memMgr.allocateMemoryOnDeviceReal(shape);
    TEST_ASSERT(realData.getData() != nullptr, "Real data should be allocated");
    TEST_ASSERT(realData.isValid(), "Real data should be valid");
    TEST_ASSERT(realData.getSize() == shape, "Real data size should match");

    // Allocate complex data (r2c output)
    ComplexData complexData = memMgr.allocateMemoryOnDeviceComplex(shape);
    TEST_ASSERT(complexData.getData() != nullptr, "Complex data should be allocated");
    TEST_ASSERT(complexData.isValid(), "Complex data should be valid");

    // Allocate real data for in-place FFT
    RealData realInPlace = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    TEST_ASSERT(realInPlace.getData() != nullptr, "Real in-place data should be allocated");
    TEST_ASSERT(realInPlace.isValid(), "Real in-place data should be valid");
    TEST_ASSERT(realInPlace.getPadding() > 0, "Real in-place data should have padding");

    // Allocate full complex data
    ComplexData complexFull = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    TEST_ASSERT(complexFull.getData() != nullptr, "Complex full data should be allocated");
    TEST_ASSERT(complexFull.isValid(), "Complex full data should be valid");
}

// ============================================================================
// Test: Data read/write via operator[]
// ============================================================================

void testDataAccess() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);

    RealData realData = memMgr.allocateMemoryOnDeviceReal(shape);
    for (int i = 0; i < shape.getVolume(); ++i) {
        realData[i] = static_cast<real_t>(i);
    }
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(realData[i], static_cast<real_t>(i)),
                    "Real data read should match written value");
    }

    ComplexData complexFull = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (int i = 0; i < shape.getVolume(); ++i) {
        complexFull[i][0] = static_cast<real_t>(i);
        complexFull[i][1] = static_cast<real_t>(i * 2);
    }
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqualComplex(complexFull[i], static_cast<real_t>(i), static_cast<real_t>(i * 2)),
                    "Complex data read should match written value");
    }
}

// ============================================================================
// Test: Complex-to-complex FFT round-trip
// ============================================================================

void testComplexFFTRoundTrip() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);

    ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    // Impulse at origin
    for (int i = 0; i < shape.getVolume(); ++i) {
        input[i][0] = 0.0f;
        input[i][1] = 0.0f;
    }
    input[0][0] = 1.0f;

    deconv.forwardFFT(input, output);
    deconv.backwardFFT(output, roundtrip);

    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqualComplex(roundtrip[i], input[i][0], input[i][1], 1e-3f),
                    "C2C FFT round-trip should recover original data");
    }
}

// ============================================================================
// Test: Real-to-complex FFT round-trip
// ============================================================================

void testRealFFTRoundTrip() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        realIn[i] = static_cast<real_t>(i % 10) * 0.1f;
    }

    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realOut);

    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(realOut[i], realIn[i], 1e-3f),
                    "R2C FFT round-trip should recover original data");
    }
}

// ============================================================================
// Test: Complex arithmetic operations
// ============================================================================

void testComplexArithmetic() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
        b[i][0] = static_cast<real_t>(i + 3);
        b[i][1] = static_cast<real_t>(i + 4);
    }

    // Complex multiplication
    deconv.complexMultiplication(a, b, result);
    for (int i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = b[i][1];
        float expectedReal = ra * rb - ia * ib;
        float expectedImag = ra * ib + ia * rb;
        TEST_ASSERT(approxEqualComplex(result[i], expectedReal, expectedImag, 1e-3f),
                    "Complex multiplication result should match expected");
    }

    // Complex addition
    deconv.complexAddition(a, b, result);
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqualComplex(result[i], a[i][0] + b[i][0], a[i][1] + b[i][1], 1e-4f),
                    "Complex addition result should match expected");
    }

    // Scalar multiplication
    complex_t scalar = {2.0f, 3.0f};
    deconv.scalarMultiplication(a, scalar, result);
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqualComplex(result[i], a[i][0] * scalar[0], a[i][1] * scalar[1], 1e-3f),
                    "Scalar multiplication result should match expected");
    }

    // Complex multiplication with conjugate
    deconv.complexMultiplicationWithConjugate(a, b, result);
    for (int i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = -b[i][1]; // conjugate
        float expectedReal = ra * rb - ia * ib;
        float expectedImag = ra * ib + ia * rb;
        TEST_ASSERT(approxEqualComplex(result[i], expectedReal, expectedImag, 1e-3f),
                    "Complex multiplication with conjugate should match expected");
    }

    // Complex division
    deconv.complexDivision(a, b, result, 1e-6f);
    for (int i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = b[i][1];
        float denom = rb * rb + ib * ib;
        if (denom < 1e-6f) {
            TEST_ASSERT(approxEqualComplex(result[i], 0.0f, 0.0f, 1e-3f),
                        "Complex division with tiny denominator should yield 0");
        } else {
            float expectedReal = (ra * rb + ia * ib) / denom;
            float expectedImag = (ia * rb - ra * ib) / denom;
            TEST_ASSERT(approxEqualComplex(result[i], expectedReal, expectedImag, 1e-2f),
                        "Complex division result should match expected");
        }
    }

    // Stabilized complex division
    deconv.complexDivisionStabilized(a, b, result, 1e-6f);
    for (int i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = b[i][1];
        float mag = std::max(1e-6f, rb * rb + ib * ib);
        float expectedReal = (ra * rb + ia * ib) / mag;
        float expectedImag = (ia * rb - ra * ib) / mag;
        TEST_ASSERT(approxEqualComplex(result[i], expectedReal, expectedImag, 1e-2f),
                    "Stabilized complex division result should match expected");
    }
}

// ============================================================================
// Test: Real arithmetic operations
// ============================================================================

void testRealArithmetic() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);

    RealData a = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData b = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData result = memMgr.allocateMemoryOnDeviceReal(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        a[i] = static_cast<real_t>(i + 1);
        b[i] = static_cast<real_t>(i + 2);
    }

    // Multiplication
    deconv.multiplication(a, b, result);
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(result[i], a[i] * b[i], 1e-3f),
                    "Real multiplication result should match expected");
    }

    // Division
    deconv.division(a, b, result, 1e-6f);
    for (int i = 0; i < shape.getVolume(); ++i) {
        real_t denom = b[i] < 1e-6f ? 1e-6f : b[i];
        TEST_ASSERT(approxEqual(result[i], a[i] / denom, 1e-2f),
                    "Real division result should match expected");
    }

    // Scalar multiplication
    real_t scalar = 2.5f;
    deconv.scalarMultiplication(a, scalar, result);
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(result[i], a[i] * scalar, 1e-3f),
                    "Real scalar multiplication result should match expected");
    }
}

// ============================================================================
// Test: Octant Fourier shift
// ============================================================================

void testOctantFourierShift() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 8);

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (int i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i);
        data[i][1] = static_cast<real_t>(i * 2);
    }

    // Double shift = identity for even sizes
    deconv.octantFourierShift(data);
    deconv.octantFourierShift(data);

    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqualComplex(data[i], static_cast<real_t>(i), static_cast<real_t>(i * 2), 1e-3f),
                    "Double octant shift should restore original data");
    }
}

// ============================================================================
// Test: Reinterpret between real and complex views
// ============================================================================

void testReinterpret() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(16, 8, 4);

    RealData realData = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    TEST_ASSERT(realData.getData() != nullptr, "Real in-place data should be allocated");

    for (int i = 0; i < shape.getVolume(); ++i) {
        realData[i] = static_cast<real_t>(i);
    }

    DataView<complex_t> complexView = memMgr.reinterpret(realData);
    TEST_ASSERT(complexView.getData() != nullptr, "Complex view should have valid data");
    TEST_ASSERT(complexView.isValid(), "Complex view should be valid");
    TEST_ASSERT(complexView.getSize().width == shape.width / 2 + 1,
                "Complex view width should be shape.width/2+1");

    DataView<real_t> realView = memMgr.reinterpret(complexView);
    TEST_ASSERT(realView.getData() != nullptr, "Real view should have valid data");
    TEST_ASSERT(realView.isValid(), "Real view should be valid");

    TEST_ASSERT(approxEqual(realView[0], 0.0f), "First element of reinterpreted view should be 0");
    TEST_ASSERT(approxEqual(realView[1], 1.0f), "Second element of reinterpreted view should be 1");
}

// ============================================================================
// Test: Memory copy
// ============================================================================

void testMemoryCopy() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);

    RealData src = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData dst = memMgr.allocateMemoryOnDeviceReal(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        src[i] = static_cast<real_t>(i * 3);
    }

    memMgr.memCopy(src, dst);

    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(dst[i], static_cast<real_t>(i * 3), 1e-4f),
                    "Copied data should match source");
    }
}

// ============================================================================
// Test: Memory tracking
// ============================================================================

void testMemoryTracking() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    size_t beforeAlloc = memMgr.getAllocatedMemory();

    {
        CuboidShape shape(32, 32, 16);
        RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
        size_t afterAlloc = memMgr.getAllocatedMemory();
        TEST_ASSERT(afterAlloc > beforeAlloc, "Allocated memory should increase after allocation");
    }

    size_t afterFree = memMgr.getAllocatedMemory();
    TEST_ASSERT(afterFree == beforeAlloc, "Allocated memory should return to baseline after free");
}

// ============================================================================
// Test: Ownership transfer
// ============================================================================

void testOwnershipTransfer() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);

    TEST_ASSERT(backend.ownsDeconvolutionBackend(), "Should own deconv backend");
    TEST_ASSERT(backend.ownsMemoryManager(), "Should own memory manager");

    // Release deconvolution backend
    auto deconvPtr = backend.releaseDeconvolutionBackend();
    TEST_ASSERT(deconvPtr != nullptr, "Released deconv backend should not be null");
    TEST_ASSERT(!backend.ownsDeconvolutionBackend(), "Should no longer own deconv backend after release");

    // Take ownership back
    backend.takeOwnership(std::move(deconvPtr));
    TEST_ASSERT(backend.ownsDeconvolutionBackend(), "Should own deconv backend after taking ownership back");

    // Release memory manager
    auto memPtr = backend.releaseMemoryManager();
    TEST_ASSERT(memPtr != nullptr, "Released memory manager should not be null");
    TEST_ASSERT(!backend.ownsMemoryManager(), "Should no longer own memory manager after release");

    // Take ownership back
    backend.takeOwnership(std::move(memPtr));
    TEST_ASSERT(backend.ownsMemoryManager(), "Should own memory manager after taking ownership back");
}

// ============================================================================
// Test: Release without ownership should throw
// ============================================================================

void testReleaseWithoutOwnershipThrows() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);

    auto deconvPtr = backend.releaseDeconvolutionBackend();

    bool threw = false;
    try {
        backend.releaseDeconvolutionBackend();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "Releasing non-owned deconv backend should throw");

    backend.takeOwnership(std::move(deconvPtr));
}

// ============================================================================
// Test: Shared memory manager access
// ============================================================================

void testSharedMemoryManager() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);

    auto sharedMem = backend.getSharedMemoryManager();
    TEST_ASSERT(sharedMem != nullptr, "Shared memory manager should not be null when owned");
    TEST_ASSERT(sharedMem.get() == backend.getMemoryManagerPtr(), "Shared memory manager should point to same object");
}

// ============================================================================
// Test: Backend clone via manager
// ============================================================================

void testBackendClone() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& original = mgr.getBackend(config);

    // CPU clone returns the same backend reference
    IBackend& cloned = mgr.clone(original, config);
    TEST_ASSERT(&cloned == &original, "CPU clone should return the same backend reference");
    TEST_ASSERT(cloned.getDeviceString() == "cpu", "Cloned backend should report 'cpu'");
}

// ============================================================================
// Test: Clone shared memory via manager
// ============================================================================

void testCloneSharedMemory() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& original = mgr.getBackend(config);
    IBackend& shared = mgr.cloneSharedMemory(original, config);

    TEST_ASSERT(shared.getDeviceString() == "cpu", "Shared memory clone should report 'cpu'");
    TEST_ASSERT(shared.hasMemoryManager(), "Shared memory clone should have a memory manager");
}

// ============================================================================
// Test: Concurrent FFT from same backend
// ============================================================================

void testConcurrentFFT() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);
    const int numThreads = 4;
    std::atomic<int> errors{0};

    auto worker = [&](int threadId) {
        try {
            ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

            for (int i = 0; i < shape.getVolume(); ++i) {
                input[i][0] = static_cast<real_t>(threadId);
                input[i][1] = 0.0f;
            }

            deconv.forwardFFT(input, output);
            deconv.backwardFFT(output, roundtrip);

            for (int i = 0; i < shape.getVolume(); ++i) {
                if (!approxEqual(roundtrip[i][0], static_cast<real_t>(threadId), 1e-3f) ||
                    !approxEqual(roundtrip[i][1], 0.0f, 1e-3f)) {
                    errors++;
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "  Thread " << threadId << " exception: " << e.what() << std::endl;
            errors++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }

    TEST_ASSERT(errors.load() == 0, "All threads should complete FFT round-trip without errors");
}

// ============================================================================
// Test: Gradient operations
// ============================================================================

void testGradientOperations() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);

    // Real data gradient
    RealData image = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gradX = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gradY = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gradZ = memMgr.allocateMemoryOnDeviceReal(shape);

    // Ramp in x: f(x,y,z) = x
    for (int z = 0; z < shape.depth; ++z) {
        for (int y = 0; y < shape.height; ++y) {
            for (int x = 0; x < shape.width; ++x) {
                image[z * shape.height * shape.width + y * shape.width + x] = static_cast<real_t>(x);
            }
        }
    }

    deconv.gradientX(image, gradX);
    // Forward diff: f(x)-f(x+1) = -1 for interior points
    for (int z = 0; z < shape.depth; ++z) {
        for (int y = 0; y < shape.height; ++y) {
            for (int x = 0; x < shape.width - 1; ++x) {
                TEST_ASSERT(approxEqual(gradX[z * shape.height * shape.width + y * shape.width + x], -1.0f, 1e-4f),
                            "X gradient of ramp should be -1");
            }
        }
    }

    // Complex data gradient
    ComplexData complexImage = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData complexGradX = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData complexGradY = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData complexGradZ = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        complexImage[i][0] = static_cast<real_t>(i);
        complexImage[i][1] = 0.0f;
    }

    deconv.gradientX(complexImage, complexGradX);
    deconv.gradientY(complexImage, complexGradY);
    deconv.gradientZ(complexImage, complexGradZ);

    bool hasNonZero = false;
    for (int i = 0; i < shape.getVolume(); ++i) {
        if (complexGradX[i][0] != 0.0f || complexGradY[i][0] != 0.0f || complexGradZ[i][0] != 0.0f) {
            hasNonZero = true;
            break;
        }
    }
    TEST_ASSERT(hasNonZero, "Complex gradients should produce some non-zero values");
}

// ============================================================================
// Test: TV normalization
// ============================================================================

void testTVNormalization() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);

    RealData gx = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gy = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gz = memMgr.allocateMemoryOnDeviceReal(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        gx[i] = 1.0f;
        gy[i] = 1.0f;
        gz[i] = 1.0f;
    }

    deconv.normalizeTV(gx, gy, gz, 1e-6f);

    // For (1,1,1), norm = sqrt(3), each component = 1/sqrt(3)
    float expected = 1.0f / std::sqrt(3.0f);
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(gx[i], expected, 1e-3f), "Normalized gx should be 1/sqrt(3)");
        TEST_ASSERT(approxEqual(gy[i], expected, 1e-3f), "Normalized gy should be 1/sqrt(3)");
        TEST_ASSERT(approxEqual(gz[i], expected, 1e-3f), "Normalized gz should be 1/sqrt(3)");
    }
}

// ============================================================================
// Test: Copy data to device
// ============================================================================

void testCopyDataToDevice() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);

    std::vector<real_t> hostData(shape.getVolume());
    for (int i = 0; i < shape.getVolume(); ++i) {
        hostData[i] = static_cast<real_t>(i * 2);
    }

    void* devicePtr = memMgr.copyDataToDevice(hostData.data(), bytes, shape);
    TEST_ASSERT(devicePtr != nullptr, "Device pointer should not be null after copy");

    std::vector<real_t> readback(shape.getVolume());
    memMgr.memCopy(devicePtr, readback.data(), bytes, shape);
    for (int i = 0; i < shape.getVolume(); ++i) {
        TEST_ASSERT(approxEqual(readback[i], hostData[i], 1e-4f), "Readback data should match host data");
    }

    memMgr.freeMemoryOnDevice(devicePtr, bytes);
}

// ============================================================================
// Test: isOnDevice
// ============================================================================

void testIsOnDevice() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    TEST_ASSERT(memMgr.isOnDevice(reinterpret_cast<const void*>(0x1)), "Non-null pointer should be 'on device' for CPU");
    TEST_ASSERT(!memMgr.isOnDevice(nullptr), "Null pointer should not be 'on device'");
}

// ============================================================================
// Test: hasNAN debug function
// ============================================================================

void testHasNAN() {
    CPUBackendManager& mgr = getManager();
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend = mgr.getBackend(config);
    IDeconvolutionBackend& deconv = backend.mutableDeconvManager();
    IBackendMemoryManager& memMgr = backend.mutableMemoryManager();

    CuboidShape shape(4, 4, 4);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i);
        data[i][1] = 0.0f;
    }

    TEST_ASSERT_NO_THROW(deconv.hasNAN(data), "hasNAN should not throw for clean data");

    data[0][0] = std::numeric_limits<real_t>::quiet_NaN();
    TEST_ASSERT_NO_THROW(deconv.hasNAN(data), "hasNAN should not throw for NaN data");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CPU Backend Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    RUN_TEST(testManagerInit);
    RUN_TEST(testBackendCreation);
    RUN_TEST(testMemoryAllocation);
    RUN_TEST(testDataAccess);
    RUN_TEST(testComplexFFTRoundTrip);
    RUN_TEST(testRealFFTRoundTrip);
    RUN_TEST(testComplexArithmetic);
    RUN_TEST(testRealArithmetic);
    RUN_TEST(testOctantFourierShift);
    RUN_TEST(testReinterpret);
    RUN_TEST(testMemoryCopy);
    RUN_TEST(testMemoryTracking);
    RUN_TEST(testOwnershipTransfer);
    RUN_TEST(testReleaseWithoutOwnershipThrows);
    RUN_TEST(testSharedMemoryManager);
    RUN_TEST(testBackendClone);
    RUN_TEST(testCloneSharedMemory);
    RUN_TEST(testConcurrentFFT);
    RUN_TEST(testGradientOperations);
    RUN_TEST(testTVNormalization);
    RUN_TEST(testCopyDataToDevice);
    RUN_TEST(testIsOnDevice);
    RUN_TEST(testHasNAN);

    std::cout << "========================================" << std::endl;
    std::cout << "  Results: " << g_testsPassed << " passed, " << g_testsFailed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return g_testsFailed > 0 ? 1 : 0;
}
