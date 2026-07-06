#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <limits>

#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include "dolphinbackend/IComputeBackend.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"

static bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool approxEqualComplex(const complex_t& a, float real, float imag, float eps = 1e-4f) {
    return approxEqual(a[0], real, eps) && approxEqual(a[1], imag, eps);
}

class CPUBackendTest : public ::testing::Test {
protected:
    static CPUBackendManager& getManager() {
        static CPUBackendManager manager;
        static bool initialized = false;
        if (!initialized) {
            manager.init([](const std::string& context, const std::string& msg, LogLevel level) {
                if (level > LogLevel::INFO){
                    std::cerr << "[" << context << "] " << msg << std::endl;
                }
            });
            initialized = true;
        }
        return manager;
    }

    BackendConfig config{nThreads: 1, backendName: "test"};
    IBackend* backend = nullptr;

    void SetUp() override {
        backend = &getManager().createBackendForCurrentThread(config);
    }
};

class CPUMemoryManagerTest : public CPUBackendTest {};
class CPUComputeBackendTest : public CPUBackendTest {};
class CPUBackendManagerTest : public CPUBackendTest {};
class CPUOwnershipTest : public CPUBackendTest {};
class CPUConcurrencyTest : public CPUBackendTest {};

// ============================================================================
// BackendManager tests
// ============================================================================

TEST_F(CPUBackendManagerTest, ManagerInit) {
    EXPECT_EQ(getManager().getNumberDevices(), 1);
}

TEST_F(CPUBackendManagerTest, GetBackendReturnsValidBackend) {
    IBackend& b = getManager().getBackend(config);
    EXPECT_EQ(b.getDeviceString(), "cpu");
    EXPECT_TRUE(b.hasMemoryManager());
    EXPECT_TRUE(b.ownsComputeBackend());
    EXPECT_TRUE(b.ownsMemoryManager());
    EXPECT_NE(b.getMemoryManagerPtr(), nullptr);
}

TEST_F(CPUBackendManagerTest, GetComputeBackendReturnsValidCompute) {
    IComputeBackend& compute = getManager().getComputeBackend(config);
    EXPECT_EQ(compute.getDeviceString(), "cpu");
}

TEST_F(CPUBackendManagerTest, GetBackendMemoryManagerReturnsValidManager) {
    IBackendMemoryManager& mem = getManager().getBackendMemoryManager(config);
    EXPECT_EQ(mem.getDeviceString(), "cpu");
}

TEST_F(CPUBackendManagerTest, MultipleBackendsAreDistinct) {
    BackendConfig config1{1, "backend1"};
    BackendConfig config2{1, "backend2"};

    IBackend& b1 = getManager().getBackend(config1);
    IBackend& b2 = getManager().getBackend(config2);

    EXPECT_NE(&b1, &b2);
    EXPECT_EQ(b1.getDeviceString(), "cpu");
    EXPECT_EQ(b2.getDeviceString(), "cpu");
    EXPECT_NE(b1.getMemoryManagerPtr(), b2.getMemoryManagerPtr());
    EXPECT_TRUE(b1.ownsComputeBackend());
    EXPECT_TRUE(b1.ownsMemoryManager());
    EXPECT_TRUE(b2.ownsComputeBackend());
    EXPECT_TRUE(b2.ownsMemoryManager());
}

TEST_F(CPUBackendManagerTest, CloneReturnsSameReference) {
    IBackend& original = getManager().getBackend(config);
    IBackend& cloned = getManager().clone(original, config);
    EXPECT_EQ(&cloned, &original);
    EXPECT_EQ(cloned.getDeviceString(), "cpu");
}

TEST_F(CPUBackendManagerTest, CloneSharedMemoryReturnsSeparateBackend) {
    IBackend& original = getManager().getBackend(config);
    IBackend& shared = getManager().cloneSharedMemory(original, config);
    EXPECT_NE(&original, &shared);
    EXPECT_EQ(shared.getDeviceString(), "cpu");
    EXPECT_TRUE(shared.hasMemoryManager());
    EXPECT_TRUE(shared.ownsComputeBackend());
    EXPECT_TRUE(shared.ownsMemoryManager());
}

TEST_F(CPUBackendManagerTest, SetThreadDistribution) {
    size_t ioThreads = 0, workerThreads = 0;
    BackendConfig ioConfig, workerConfig;
    getManager().setThreadDistribution(4, ioThreads, workerThreads, ioConfig, workerConfig);
    EXPECT_GE(ioThreads + workerThreads, 1u);
}

TEST_F(CPUBackendManagerTest, DifferentThreadConfigsProduceSameResults) {
    BackendConfig config1{1, "single_thread"};
    BackendConfig config2{4, "multi_thread"};

    IBackend& b1 = getManager().getBackend(config1);
    IBackend& b2 = getManager().getBackend(config2);

    IComputeBackend& d1 = b1.mutableComputeManager();
    IComputeBackend& d2 = b2.mutableComputeManager();
    IBackendMemoryManager& m1 = b1.mutableMemoryManager();
    IBackendMemoryManager& m2 = b2.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);
    ComplexData in1 = m1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData out1 = m1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData rt1 = m1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData in2 = m2.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData out2 = m2.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData rt2 = m2.allocateMemoryOnDeviceComplexFull(shape);

    ASSERT_TRUE(in1.isValid()) << "in1 allocation failed";
    ASSERT_TRUE(out1.isValid()) << "out1 allocation failed";
    ASSERT_TRUE(rt1.isValid()) << "rt1 allocation failed";
    ASSERT_TRUE(in2.isValid()) << "in2 allocation failed";
    ASSERT_TRUE(out2.isValid()) << "out2 allocation failed";
    ASSERT_TRUE(rt2.isValid()) << "rt2 allocation failed";

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        in1[i][0] = 1.0f; in1[i][1] = 0.0f;
        in2[i][0] = 1.0f; in2[i][1] = 0.0f;
    }

    try {
        d1.forwardFFT(in1, out1);
    } catch (const std::exception& e) {
        FAIL() << "d1.forwardFFT threw: " << e.what();
    }

    try {
        d1.backwardFFT(out1, rt1);
    } catch (const std::exception& e) {
        FAIL() << "d1.backwardFFT threw: " << e.what();
    }

    try {
        d2.forwardFFT(in2, out2);
    } catch (const std::exception& e) {
        FAIL() << "d2.forwardFFT threw: " << e.what();
    }

    try {
        d2.backwardFFT(out2, rt2);
    } catch (const std::exception& e) {
        FAIL() << "d2.backwardFFT threw: " << e.what();
    }

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(rt1[i][0], rt2[i][0], 1e-3f) << " mismatch at index " << i << " real";
        EXPECT_NEAR(rt1[i][1], rt2[i][1], 1e-3f) << " mismatch at index " << i << " imag";
    }
}

// ============================================================================
// BackendMemoryManager tests
// ============================================================================

TEST_F(CPUMemoryManagerTest, AllocateReal) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
    EXPECT_EQ(data.getSize(), shape);
}

TEST_F(CPUMemoryManagerTest, AllocateComplex) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplex(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
}

TEST_F(CPUMemoryManagerTest, AllocateRealFFTInPlace) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    RealData data = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
    EXPECT_GT(data.getPadding(), 0u);
}

TEST_F(CPUMemoryManagerTest, AllocateComplexFull) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
}

TEST_F(CPUMemoryManagerTest, DataAccessReal) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i] = static_cast<real_t>(i);
    }
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<real_t>(i));
    }
}

TEST_F(CPUMemoryManagerTest, DataAccessComplex) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i);
        data[i][1] = static_cast<real_t>(i * 2);
    }
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_FLOAT_EQ(data[i][0], static_cast<real_t>(i));
        EXPECT_FLOAT_EQ(data[i][1], static_cast<real_t>(i * 2));
    }
}

TEST_F(CPUMemoryManagerTest, MemoryTrackingIncreasesOnAlloc) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    size_t before = memMgr.getAllocatedMemory();
    {
        CuboidShape shape(32, 32, 16);
        RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
        EXPECT_GT(memMgr.getAllocatedMemory(), before);
    }
    EXPECT_EQ(memMgr.getAllocatedMemory(), before);
}

TEST_F(CPUMemoryManagerTest, SharedMemoryTrackingAcrossBackends) {
    BackendConfig cfg{1, "tracking_test"};
    IBackend& b1 = getManager().getBackend(config);
    IBackend& b2 = getManager().getBackend(cfg);
    IBackendMemoryManager& m1 = b1.mutableMemoryManager();
    IBackendMemoryManager& m2 = b2.mutableMemoryManager();

    size_t baseline = m1.getAllocatedMemory();
    CuboidShape shape(16, 16, 8);
    RealData data = m2.allocateMemoryOnDeviceReal(shape);
    EXPECT_GT(m1.getAllocatedMemory(), baseline);
}

TEST_F(CPUMemoryManagerTest, MemoryCopy) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    RealData src = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData dst = memMgr.allocateMemoryOnDeviceReal(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        src[i] = static_cast<real_t>(i * 3);
    }
    memMgr.memCopy(src, dst);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_FLOAT_EQ(dst[i], static_cast<real_t>(i * 3));
    }
}

TEST_F(CPUMemoryManagerTest, CopyDataToDevice) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);
    std::vector<real_t> hostData(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostData[i] = static_cast<real_t>(i * 2);
    }
    void* devicePtr = memMgr.copyDataToDevice(hostData.data(), bytes, shape);
    EXPECT_NE(devicePtr, nullptr);
    std::vector<real_t> readback(shape.getVolume());
    memMgr.memCopy(devicePtr, readback.data(), bytes, shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_FLOAT_EQ(readback[i], hostData[i]);
    }
    memMgr.freeMemoryOnDevice(devicePtr, bytes);
}

TEST_F(CPUMemoryManagerTest, MoveDataBetweenBackends) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);
    RealData src = memMgr.allocateMemoryOnDeviceReal(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        src[i] = static_cast<real_t>(i * 5);
    }
    void* movedPtr = memMgr.moveDataFromDevice(src.getData(), bytes, shape, memMgr);
    EXPECT_NE(movedPtr, nullptr);
    EXPECT_EQ(movedPtr, src.getData());
    src.setData(nullptr);
    memMgr.freeMemoryOnDevice(movedPtr, bytes);
}

TEST_F(CPUMemoryManagerTest, ReinterpretRealToComplex) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 8, 4);
    RealData realData = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    EXPECT_NE(realData.getData(), nullptr);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        realData[i] = static_cast<real_t>(i);
    }
    DataView<complex_t> complexView = memMgr.reinterpret(realData);
    EXPECT_NE(complexView.getData(), nullptr);
    EXPECT_TRUE(complexView.isValid());
    EXPECT_EQ(complexView.getSize().width, shape.width / 2 + 1);
}

TEST_F(CPUMemoryManagerTest, ReinterpretComplexToReal) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 8, 4);
    RealData realData = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        realData[i] = static_cast<real_t>(i);
    }
    DataView<complex_t> complexView = memMgr.reinterpret(realData);
    DataView<real_t> realView = memMgr.reinterpret(complexView);
    EXPECT_NE(realView.getData(), nullptr);
    EXPECT_TRUE(realView.isValid());
    EXPECT_FLOAT_EQ(realView[0], 0.0f);
    EXPECT_FLOAT_EQ(realView[1], 1.0f);
}

TEST_F(CPUMemoryManagerTest, IsOnDevice) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_TRUE(memMgr.isOnDevice(reinterpret_cast<const void*>(0x1)));
    EXPECT_FALSE(memMgr.isOnDevice(nullptr));
}

TEST_F(CPUMemoryManagerTest, GetAllocatedMemory) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_NO_THROW({
        size_t mem = memMgr.getAllocatedMemory();
        (void)mem;
    });
}

TEST_F(CPUMemoryManagerTest, FreeMemoryOnDevice) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    void* ptr = memMgr.allocateMemoryOnDevice(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(memMgr.freeMemoryOnDevice(ptr, 1024));
}

TEST_F(CPUMemoryManagerTest, CreateCopy) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    RealData src = memMgr.allocateMemoryOnDeviceReal(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        src[i] = static_cast<real_t>(i * 7);
    }
    RealData copy = memMgr.createCopy(src);
    EXPECT_NE(copy.getData(), nullptr);
    EXPECT_TRUE(copy.isValid());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_FLOAT_EQ(copy[i], src[i]);
    }
}

TEST_F(CPUMemoryManagerTest, Sync) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_NO_THROW(memMgr.sync());
}

// ============================================================================
// ComputeBackend tests
// ============================================================================

TEST_F(CPUComputeBackendTest, ComplexFFTRoundTrip) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        input[i][0] = 0.0f;
        input[i][1] = 0.0f;
    }
    input[0][0] = 1.0f;

    deconv.forwardFFT(input, output);
    deconv.backwardFFT(output, roundtrip);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqualComplex(roundtrip[i], input[i][0], input[i][1], 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, RealFFTRoundTrip) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        realIn[i] = static_cast<real_t>(i % 10) * 0.1f;
    }

    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realOut);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(realOut[i], realIn[i], 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, ComplexMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
        b[i][0] = static_cast<real_t>(i + 3);
        b[i][1] = static_cast<real_t>(i + 4);
    }

    deconv.complexMultiplication(a, b, result);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = b[i][1];
        EXPECT_TRUE(approxEqualComplex(result[i], ra*rb - ia*ib, ra*ib + ia*rb, 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, ComplexAddition) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
        b[i][0] = static_cast<real_t>(i + 3);
        b[i][1] = static_cast<real_t>(i + 4);
    }

    deconv.complexAddition(a, b, result);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqualComplex(result[i], a[i][0]+b[i][0], a[i][1]+b[i][1], 1e-4f));
    }
}

TEST_F(CPUComputeBackendTest, ComplexScalarMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
    }

    complex_t scalar = {2.0f, 3.0f};
    deconv.scalarMultiplication(a, scalar, result);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        EXPECT_TRUE(approxEqualComplex(result[i], ra*scalar[0]-ia*scalar[1], ra*scalar[1]+ia*scalar[0], 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, ComplexMultiplicationWithConjugate) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
        b[i][0] = static_cast<real_t>(i + 3);
        b[i][1] = static_cast<real_t>(i + 4);
    }

    deconv.complexMultiplicationWithConjugate(a, b, result);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = -b[i][1];
        EXPECT_TRUE(approxEqualComplex(result[i], ra*rb-ia*ib, ra*ib+ia*rb, 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, ComplexDivision) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
        b[i][0] = static_cast<real_t>(i + 3);
        b[i][1] = static_cast<real_t>(i + 4);
    }

    deconv.complexDivision(a, b, result, 1e-6f);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = b[i][1];
        float denom = rb*rb + ib*ib;
        if (denom < 1e-6f) {
            EXPECT_TRUE(approxEqualComplex(result[i], 0.0f, 0.0f, 1e-3f));
        } else {
            EXPECT_TRUE(approxEqualComplex(result[i], (ra*rb+ia*ib)/denom, (ia*rb-ra*ib)/denom, 1e-2f));
        }
    }
}

TEST_F(CPUComputeBackendTest, ComplexDivisionStabilized) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i][0] = static_cast<real_t>(i + 1);
        a[i][1] = static_cast<real_t>(i + 2);
        b[i][0] = static_cast<real_t>(i + 3);
        b[i][1] = static_cast<real_t>(i + 4);
    }

    deconv.complexDivisionStabilized(a, b, result, 1e-6f);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = a[i][0], ia = a[i][1];
        real_t rb = b[i][0], ib = b[i][1];
        float mag = std::max(1e-6f, rb*rb + ib*ib);
        EXPECT_TRUE(approxEqualComplex(result[i], (ra*rb+ia*ib)/mag, (ia*rb-ra*ib)/mag, 1e-2f));
    }
}

TEST_F(CPUComputeBackendTest, RealMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData a = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData b = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData result = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i] = static_cast<real_t>(i + 1);
        b[i] = static_cast<real_t>(i + 2);
    }

    deconv.multiplication(a, b, result);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(result[i], a[i] * b[i], 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, RealDivision) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData a = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData b = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData result = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i] = static_cast<real_t>(i + 1);
        b[i] = static_cast<real_t>(i + 2);
    }

    deconv.division(a, b, result, 1e-6f);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t denom = b[i] < 1e-6f ? 1e-6f : b[i];
        EXPECT_TRUE(approxEqual(result[i], a[i] / denom, 1e-2f));
    }
}

TEST_F(CPUComputeBackendTest, RealScalarMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData a = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData result = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        a[i] = static_cast<real_t>(i + 1);
    }

    real_t scalar = 2.5f;
    deconv.scalarMultiplication(a, scalar, result);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(result[i], a[i] * scalar, 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, OctantFourierShiftDoubleIsIdentity) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 8);

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i);
        data[i][1] = static_cast<real_t>(i * 2);
    }

    deconv.octantFourierShift(data);
    deconv.octantFourierShift(data);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqualComplex(data[i], static_cast<real_t>(i), static_cast<real_t>(i * 2), 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, GradientX) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData image = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gradX = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t z = 0; z < shape.depth; ++z)
        for (size_t y = 0; y < shape.height; ++y)
            for (size_t x = 0; x < shape.width; ++x)
                image[z * shape.height * shape.width + y * shape.width + x] = static_cast<real_t>(x);

    deconv.gradientX(image, gradX);
    for (size_t z = 0; z < shape.depth; ++z)
        for (size_t y = 0; y < shape.height; ++y)
            for (size_t x = 0; x < shape.width - 1; ++x)
                EXPECT_TRUE(approxEqual(gradX[z * shape.height * shape.width + y * shape.width + x], -1.0f, 1e-4f));
}

TEST_F(CPUComputeBackendTest, ComplexGradients) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData image = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData gradX = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData gradY = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData gradZ = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        image[i][0] = static_cast<real_t>(i);
        image[i][1] = 0.0f;
    }

    deconv.gradientX(image, gradX);
    deconv.gradientY(image, gradY);
    deconv.gradientZ(image, gradZ);

    bool hasNonZero = false;
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        if (gradX[i][0] != 0.0f || gradY[i][0] != 0.0f || gradZ[i][0] != 0.0f) {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero);
}

TEST_F(CPUComputeBackendTest, TVNormalization) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData gx = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gy = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gz = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        gx[i] = 1.0f;
        gy[i] = 1.0f;
        gz[i] = 1.0f;
    }

    deconv.normalizeTV(gx, gy, gz, 1e-6f);
    float expected = 1.0f / std::sqrt(3.0f);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(gx[i], expected, 1e-3f));
        EXPECT_TRUE(approxEqual(gy[i], expected, 1e-3f));
        EXPECT_TRUE(approxEqual(gz[i], expected, 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, HasNANDoesNotThrow) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i);
        data[i][1] = 0.0f;
    }
    EXPECT_NO_THROW(deconv.hasNAN(data));

    data[0][0] = std::numeric_limits<real_t>::quiet_NaN();
    EXPECT_NO_THROW(deconv.hasNAN(data));
}

// ============================================================================
// Ownership tests
// ============================================================================

TEST_F(CPUOwnershipTest, TransferComputeBackend) {
    EXPECT_TRUE(backend->ownsComputeBackend());
    auto deconvPtr = backend->releaseComputeBackend();
    EXPECT_NE(deconvPtr, nullptr);
    EXPECT_FALSE(backend->ownsComputeBackend());
    backend->takeOwnership(std::move(deconvPtr));
    EXPECT_TRUE(backend->ownsComputeBackend());
}

TEST_F(CPUOwnershipTest, TransferMemoryManager) {
    EXPECT_TRUE(backend->ownsMemoryManager());
    auto memPtr = backend->releaseMemoryManager();
    EXPECT_NE(memPtr, nullptr);
    EXPECT_FALSE(backend->ownsMemoryManager());
    backend->takeOwnership(std::move(memPtr));
    EXPECT_TRUE(backend->ownsMemoryManager());
}

TEST_F(CPUOwnershipTest, ReleaseWithoutOwnershipThrows) {
    auto deconvPtr = backend->releaseComputeBackend();
    EXPECT_THROW(backend->releaseComputeBackend(), std::runtime_error);
    backend->takeOwnership(std::move(deconvPtr));
}

TEST_F(CPUOwnershipTest, TakeOwnershipAlreadyOwnedThrows) {
    EXPECT_THROW({
        auto deconvPtr = backend->releaseComputeBackend();
        backend->takeOwnership(std::move(deconvPtr));
        auto deconvPtr2 = backend->releaseComputeBackend();
        backend->takeOwnership(std::move(deconvPtr2));
        backend->takeOwnership(std::move(deconvPtr2));
    }, std::runtime_error);
}

TEST_F(CPUOwnershipTest, SharedMemoryManager) {
    auto sharedMem = backend->getSharedMemoryManager();
    EXPECT_NE(sharedMem, nullptr);
    EXPECT_EQ(sharedMem.get(), backend->getMemoryManagerPtr());
}

// ============================================================================
// Concurrency tests
// ============================================================================

TEST_F(CPUConcurrencyTest, ConcurrentFFTSameBackend) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    const int numThreads = 4;
    std::atomic<int> errors{0};

    auto worker = [&](int threadId) {
        try {
            ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

            for (size_t i = 0; i < shape.getVolume(); ++i) {
                input[i][0] = static_cast<real_t>(threadId);
                input[i][1] = 0.0f;
            }
            deconv.forwardFFT(input, output);
            deconv.backwardFFT(output, roundtrip);

            for (size_t i = 0; i < shape.getVolume(); ++i) {
                if (!approxEqual(roundtrip[i][0], static_cast<real_t>(threadId), 1e-3f) ||
                    !approxEqual(roundtrip[i][1], 0.0f, 1e-3f)) {
                    errors++;
                    break;
                }
            }
        } catch (...) {
            errors++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i)
        threads.emplace_back(worker, i);
    for (auto& t : threads)
        t.join();

    EXPECT_EQ(errors.load(), 0);
}

TEST_F(CPUConcurrencyTest, ConcurrentDifferentBackends) {
    BackendConfig cfg{1, "concurrent_test"};
    IBackend& backend2 = getManager().getBackend(cfg);

    IComputeBackend& d1 = backend->mutableComputeManager();
    IComputeBackend& d2 = backend2.mutableComputeManager();
    IBackendMemoryManager& m1 = backend->mutableMemoryManager();
    IBackendMemoryManager& m2 = backend2.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);
    std::atomic<int> errors{0};

    auto worker = [&](IComputeBackend& deconv, IBackendMemoryManager& memMgr, float value) {
        try {
            ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

            for (size_t i = 0; i < shape.getVolume(); ++i) {
                input[i][0] = value;
                input[i][1] = 0.0f;
            }
            deconv.forwardFFT(input, output);
            deconv.backwardFFT(output, roundtrip);

            for (size_t i = 0; i < shape.getVolume(); ++i) {
                if (!approxEqual(roundtrip[i][0], value, 1e-3f) ||
                    !approxEqual(roundtrip[i][1], 0.0f, 1e-3f)) {
                    errors++;
                    break;
                }
            }
        } catch (...) {
            errors++;
        }
    };

    std::thread t1(worker, std::ref(d1), std::ref(m1), 1.0f);
    std::thread t2(worker, std::ref(d2), std::ref(m2), 2.0f);
    t1.join();
    t2.join();

    EXPECT_EQ(errors.load(), 0);
}
