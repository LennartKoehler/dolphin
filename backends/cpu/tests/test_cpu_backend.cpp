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
#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <limits>
#include <utility>

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
                if (level > LogLevel::LOG_INFO){
                    std::cerr << "[" << context << "] " << msg << std::endl;
                }
            });
            initialized = true;
        }
        return manager;
    }

    BackendConfig config{1, "test"};
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
    IBackend& b = getManager().createBackendForCurrentThread(config);
    EXPECT_EQ(b.getDeviceString(), "cpu");
    EXPECT_TRUE(b.hasMemoryManager());
    EXPECT_TRUE(b.ownsComputeBackend());
    EXPECT_TRUE(b.ownsMemoryManager());
    EXPECT_NE(b.getMemoryManagerPtr(), nullptr);
}

// TEST_F(CPUBackendManagerTest, GetComputeBackendReturnsValidCompute) {
//     IComputeBackend& compute = getManager().getComputeBackend(config);
//     EXPECT_EQ(compute.getDeviceString(), "cpu");
// }

// TEST_F(CPUBackendManagerTest, GetBackendMemoryManagerReturnsValidManager) {
//     IBackendMemoryManager& mem = getManager().getBackendMemoryManager(config);
//     EXPECT_EQ(mem.getDeviceString(), "cpu");
// }

TEST_F(CPUBackendManagerTest, MultipleBackendsAreDistinct) {
    BackendConfig config1{1, "backend1"};
    BackendConfig config2{1, "backend2"};

    IBackend& b1 = getManager().createBackendForCurrentThread(config1);
    IBackend& b2 = getManager().createBackendForCurrentThread(config2);

    EXPECT_NE(&b1, &b2);
    EXPECT_EQ(b1.getDeviceString(), "cpu");
    EXPECT_EQ(b2.getDeviceString(), "cpu");
    EXPECT_NE(b1.getMemoryManagerPtr(), b2.getMemoryManagerPtr());
    EXPECT_TRUE(b1.ownsComputeBackend());
    EXPECT_TRUE(b1.ownsMemoryManager());
    EXPECT_TRUE(b2.ownsComputeBackend());
    EXPECT_TRUE(b2.ownsMemoryManager());
}

// TEST_F(CPUBackendManagerTest, CloneReturnsSameReference) {
//     IBackend& original = getManager().getBackend(config);
//     IBackend& cloned = getManager().clone(original, config);
//     EXPECT_EQ(&cloned, &original);
//     EXPECT_EQ(cloned.getDeviceString(), "cpu");
// }

TEST_F(CPUBackendManagerTest, CloneSharedMemoryReturnsSeparateBackend) {
    IBackend& original = getManager().createBackendForCurrentThread(config);
    IBackend& shared = getManager().createBackendSharedMemoryForCurrentThread(original, config);
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

    IBackend& b1 = getManager().createBackendForCurrentThread(config1);
    IBackend& b2 = getManager().createBackendForCurrentThread(config2);

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
    IBackend& b1 = getManager().createBackendForCurrentThread(config);
    IBackend& b2 = getManager().createBackendForCurrentThread(cfg);
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

TEST_F(CPUComputeBackendTest, RealFFTRoundTripOutOfPlaceUnpadded) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceReal(shape);
    ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        realIn[i] = static_cast<real_t>(i % 10) * 0.1f;
    }

    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realOut);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(realOut[i], realIn[i], 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, RealFFTMixedPadding) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        realIn[i] = static_cast<real_t>(i % 7) * 0.2f;
    }

    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realOut);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(realOut[i], realIn[i], 1e-3f));
    }
}

TEST_F(CPUComputeBackendTest, ForwardFFTRejectsFullComplexOutput) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceReal(shape);
    ComplexData fullOut = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    EXPECT_THROW(deconv.forwardFFT(realIn, fullOut), dolphin::backend::BackendException);
}

TEST_F(CPUComputeBackendTest, BackwardFFTRejectsFullComplexInput) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    ComplexData fullIn = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);

    EXPECT_THROW(deconv.backwardFFT(fullIn, realOut), dolphin::backend::BackendException);
}

// ---------------------------------------------------------------------------
// Exact-reference FFT tests (4x4x4).
//
// Reference values are precomputed externally with a direct 3D DFT matching the
// backend convention (forward unnormalized, backward normalized by 1/64), i.e.
// the same definition as numpy fftn/ifftn/rfftn/irfftn. Element order is
// (depth, height, width) with width fastest. Each direction is validated
// independently against its own precomputed input->expected pair (no
// forward-then-backward chaining), so a wrong sign, axis order, or
// normalization constant is caught - things a self round-trip would hide.
// kRealInput == kRealFwdRef's inverse (c2r expected output), kComplexInput ==
// kComplexFwdRef's inverse (c2c backward expected output).
// ---------------------------------------------------------------------------
// 64 values, order (d,h,w), w fastest
static const float kRealInput[64] = {
    1.5f, 2.144218f, 2.48545f, 2.363209f, 1.67052f, 2.216471f, 2.366665f, 2.050463f,
    1.814642f, 2.213558f, 2.159297f, 1.67738f, 1.908327f, 2.124574f, 1.870705f, 1.26612f,
    2.229426f, 2.682039f, 2.6963f, 2.265501f, 2.342356f, 2.622495f, 2.433496f, 1.864249f,
    2.391207f, 2.473848f, 2.098472f, 1.441626f, 2.36045f, 2.238209f, 1.709988f, 1.024217f,
    2.841471f, 2.991665f, 2.675463f, 2.041581f, 2.838558f, 2.784297f, 2.30238f, 1.619459f,
    2.749574f, 2.495705f, 1.89112f, 1.220164f, 2.5713f, 2.140501f, 1.467254f, 0.8681975f,
    3.247495f, 3.058496f, 2.489249f, 1.80748f, 3.098848f, 2.723472f, 2.066626f, 1.437234f,
    2.863209f, 2.334988f, 1.649217f, 1.128424f, 2.550463f, 1.916581f, 1.263142f, 0.8974699f,
};

// 48 values (width/2+1), order (d,h,w), w fastest
static const std::pair<float, float> kRealFwdRef[48] = {
    { 136.7366f, 0.0f }, { 5.353021f, -14.18834f }, { 8.46878f, 0.0f }, { 6.916611f, -8.260091f }, { -1.560067f, 3.950193f }, { -1.820381f, 1.424452f }, { 7.506386f, 0.0f }, { -2.36811f, 1.290039f },
    { -1.585358f, 0.0f }, { 6.916611f, 8.260091f }, { -3.537128f, -0.6464766f }, { -1.820381f, -1.424452f }, { -3.66709f, -0.3414858f }, { -3.311381f, 7.227719f }, { -3.455937f, 2.274593f }, { -0.7968273f, 2.521816f },
    { 1.783271f, 0.1766f }, { 0.360273f, 0.7431893f }, { -1.430885f, 0.9417652f }, { 0.6198914f, 0.8819139f }, { -0.2390408f, 0.4601886f }, { -2.334409f, -0.204279f }, { -0.359695f, 1.290551f }, { -0.6879596f, 0.09236156f },
    { -2.075984f, 0.0f }, { -4.000905f, 2.179512f }, { -2.678451f, 0.0f }, { -1.273379f, 0.9964219f }, { 0.4934078f, 0.8143286f }, { -0.2718488f, 0.3970706f }, { -1.108977f, 0.0f }, { -0.2065793f, 0.5475446f },
    { -0.32682f, 0.0f }, { -1.273379f, -0.9964219f }, { -0.7291756f, 0.2044635f }, { -0.2718488f, -0.3970706f }, { -3.66709f, 0.3414858f }, { -5.995245f, -0.5854793f }, { -3.455937f, -2.274593f }, { -2.334409f, 0.204279f },
    { -0.5100081f, 1.014623f }, { -0.6879596f, -0.09236156f }, { -1.430885f, -0.9417652f }, { -0.7315295f, 0.1096019f }, { -0.2390408f, -0.4601886f }, { -0.7968273f, -2.521816f }, { -0.7069638f, -0.7629233f }, { 0.360273f, -0.7431893f },
};

// 64 values, order (d,h,w), w fastest
static const std::pair<float, float> kComplexInput[64] = {
    { 1.f, 1.f }, { 0.721061f, 0.7f }, { 0.2967067f, 0.4f }, { -0.2376422f, 0.1f }, { 1.7f, 2.391207f }, { 1.421061f, 2.091207f }, { 0.9967067f, 1.791207f }, { 0.4623578f, 1.491207f },
    { 2.4f, 2.808496f }, { 2.121061f, 2.508496f }, { 1.696707f, 2.208496f }, { 1.162358f, 1.908496f }, { 3.1f, 2.342254f }, { 2.821061f, 2.042254f }, { 2.396707f, 1.742254f }, { 1.862358f, 1.442254f },
    { 0.92161f, 1.564642f }, { 0.3674988f, 1.264642f }, { -0.2288445f, 0.9646425f }, { -0.8048461f, 0.6646425f }, { 1.62161f, 2.491665f }, { 1.067499f, 2.191665f }, { 0.4711555f, 1.891665f }, { -0.1048461f, 1.591665f },
    { 2.32161f, 2.334988f }, { 1.767499f, 2.034988f }, { 1.171156f, 1.734988f }, { 0.5951539f, 1.434988f }, { 3.02161f, 1.812234f }, { 2.467499f, 1.512234f }, { 1.871156f, 1.212234f }, { 1.295154f, 0.9122338f },
    { 0.3727979f, 1.932039f }, { -0.1885011f, 1.632039f }, { -0.6568888f, 1.332039f }, { -0.9899925f, 1.032039f }, { 1.072798f, 2.245705f }, { 0.5114989f, 1.945705f }, { 0.04311125f, 1.645705f }, { -0.2899925f, 1.345705f },
    { 1.772798f, 1.744459f }, { 1.211499f, 1.444459f }, { 0.7431112f, 1.144459f }, { 0.4100075f, 0.8444589f }, { 2.472798f, 1.52247f }, { 1.911499f, 1.22247f }, { 1.443111f, 0.9224699f }, { 1.110008f, 0.6224699f },
    { -0.004072142f, 1.973848f }, { -0.2991352f, 1.673848f }, { -0.4364567f, 1.373848f }, { -0.4259323f, 1.073848f }, { 0.6959279f, 1.739249f }, { 0.4008648f, 1.439249f }, { 0.2635433f, 1.139249f }, { 0.2740677f, 0.8392493f },
    { 1.395928f, 1.243198f }, { 1.100865f, 0.9431975f }, { 0.9635433f, 0.6431975f }, { 0.9740677f, 0.3431975f }, { 2.095928f, 1.574185f }, { 1.800865f, 1.274185f }, { 1.663543f, 0.9741853f }, { 1.674068f, 0.6741853f },
};

// 64 values, order (d,h,w), w fastest
static const std::pair<float, float> kComplexFwdRef[64] = {
    { 64.82945f, 94.08256f }, { 22.86328f, -2.637347f }, { 12.48937f, 9.6f }, { 3.663276f, 21.83735f }, { -15.93327f, 15.75755f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { -22.4f, -6.0692f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { -28.86673f, -29.04245f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { 19.66304f, -1.294918f }, { -5.487765f, -3.501127f }, { -0.3244592f, -3.382296f }, { 2.876618f, -2.243432f }, { -8.701112f, -10.04177f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { -1.231514f, -3.333153f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { -3.306854f, -5.926839f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { 4.910876f, 5.010488f }, { 0.6005638f, -1.84421f }, { 1.172157f, 0.0f }, { 0.6005638f, 1.84421f }, { -0.2892264f, -6.324883f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { 0.0f, -2.063939f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.2892264f, -6.324883f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { 6.278642f, 10.0732f }, { 2.876618f, 2.243432f }, { -0.3244592f, 3.382296f }, { -5.487765f, 3.501127f }, { 3.306854f, -5.926839f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
    { 1.231514f, -3.333153f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 8.701112f, -10.04177f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
};

TEST_F(CPUComputeBackendTest, RealForwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place: unpadded real input -> complex output
    {
        RealData realIn = memMgr.allocateMemoryOnDeviceReal(shape);
        ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
        for (size_t i = 0; i < 64; ++i) realIn[i] = kRealInput[i];
        deconv.forwardFFT(realIn, complexOut);
        for (size_t i = 0; i < 48; ++i) {
            EXPECT_NEAR(complexOut[i][0], kRealFwdRef[i].first, tol) << "OOP fwd real bin " << i;
            EXPECT_NEAR(complexOut[i][1], kRealFwdRef[i].second, tol) << "OOP fwd real bin " << i;
        }
    }
    // In-place: padded real buffer reinterpreted as a complex view
    {
        RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
        DataView<complex_t> complexOut = memMgr.reinterpret(realIn);
        for (size_t i = 0; i < 64; ++i) realIn[i] = kRealInput[i];
        deconv.forwardFFT(realIn, complexOut);
        for (size_t i = 0; i < 48; ++i) {
            EXPECT_NEAR(complexOut[i][0], kRealFwdRef[i].first, tol) << "IP fwd real bin " << i;
            EXPECT_NEAR(complexOut[i][1], kRealFwdRef[i].second, tol) << "IP fwd real bin " << i;
        }
    }
}

TEST_F(CPUComputeBackendTest, RealBackwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place: complex input -> unpadded real output
    {
        ComplexData complexIn = memMgr.allocateMemoryOnDeviceComplex(shape);
        RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);
        for (size_t i = 0; i < 48; ++i) {
            complexIn[i][0] = kRealFwdRef[i].first;
            complexIn[i][1] = kRealFwdRef[i].second;
        }
        deconv.backwardFFT(complexIn, realOut);
        for (size_t i = 0; i < 64; ++i)
            EXPECT_NEAR(realOut[i], kRealInput[i], tol) << "OOP bwd real bin " << i;
    }
    // In-place: complex view on a padded real buffer
    {
        RealData realOut = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
        DataView<complex_t> complexIn = memMgr.reinterpret(realOut);
        for (size_t i = 0; i < 48; ++i) {
            complexIn[i][0] = kRealFwdRef[i].first;
            complexIn[i][1] = kRealFwdRef[i].second;
        }
        deconv.backwardFFT(complexIn, realOut);
        for (size_t i = 0; i < 64; ++i)
            EXPECT_NEAR(realOut[i], kRealInput[i], tol) << "IP bwd real bin " << i;
    }
}

TEST_F(CPUComputeBackendTest, ComplexForwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place
    {
        ComplexData in = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        ComplexData out = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        for (size_t i = 0; i < 64; ++i) {
            in[i][0] = kComplexInput[i].first;
            in[i][1] = kComplexInput[i].second;
        }
        deconv.forwardFFT(in, out);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(out[i][0], kComplexFwdRef[i].first, tol) << "OOP fwd complex bin " << i;
            EXPECT_NEAR(out[i][1], kComplexFwdRef[i].second, tol) << "OOP fwd complex bin " << i;
        }
    }
    // In-place
    {
        ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        for (size_t i = 0; i < 64; ++i) {
            data[i][0] = kComplexInput[i].first;
            data[i][1] = kComplexInput[i].second;
        }
        deconv.forwardFFT(data, data);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(data[i][0], kComplexFwdRef[i].first, tol) << "IP fwd complex bin " << i;
            EXPECT_NEAR(data[i][1], kComplexFwdRef[i].second, tol) << "IP fwd complex bin " << i;
        }
    }
}

TEST_F(CPUComputeBackendTest, ComplexBackwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place
    {
        ComplexData in = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        ComplexData out = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        for (size_t i = 0; i < 64; ++i) {
            in[i][0] = kComplexFwdRef[i].first;
            in[i][1] = kComplexFwdRef[i].second;
        }
        deconv.backwardFFT(in, out);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(out[i][0], kComplexInput[i].first, tol) << "OOP bwd complex bin " << i;
            EXPECT_NEAR(out[i][1], kComplexInput[i].second, tol) << "OOP bwd complex bin " << i;
        }
    }
    // In-place
    {
        ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        for (size_t i = 0; i < 64; ++i) {
            data[i][0] = kComplexFwdRef[i].first;
            data[i][1] = kComplexFwdRef[i].second;
        }
        deconv.backwardFFT(data, data);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(data[i][0], kComplexInput[i].first, tol) << "IP bwd complex bin " << i;
            EXPECT_NEAR(data[i][1], kComplexInput[i].second, tol) << "IP bwd complex bin " << i;
        }
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

// ============================================================================
// octantFourierShift — regression tests
//
// Semantics (verified against PSFHandler.cpp:183 usage):
//   Swaps element (z,y,x) with ((z+halfD)%D, (y+halfH)%H, (x+halfW)%W)
//   for z in [0, halfD), y in [0, H), x in [0, W).
//   Because z iterates only the lower half, each pair is swapped exactly once
//   (the z-partition guarantees no double-swap in x/y).  The operation is an
//   involution for ALL dimensions.  For odd dims the last z-slice (z=D-1) is
//   a fixed point — this is benign for zero-padded PSFs and is NOT a bug.
//   A peak at floor(N/2) lands at the origin for both even and odd dims.
// ============================================================================

// Reference implementation operating on a flat host buffer (logical indices).
template <typename T>
static void referenceShift(std::vector<T>& buf, size_t W, size_t H, size_t D) {
    size_t halfW = W / 2, halfH = H / 2, halfD = D / 2;
    for (size_t z = 0; z < halfD; ++z)
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x) {
                size_t i1 = z * H * W + y * W + x;
                size_t i2 = ((z + halfD) % D) * H * W +
                            ((y + halfH) % H) * W +
                            ((x + halfW) % W);
                if (i1 != i2) std::swap(buf[i1], buf[i2]);
            }
}

TEST_F(CPUComputeBackendTest, OctantShiftMovesCenterToOrigin) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{8, 8, 8};

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = 0.0f;
        data[i][1] = 0.0f;
    }
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    data[centerIdx][0] = 42.0f;
    data[centerIdx][1] = 7.0f;
    data[0][0] = 1.0f;
    data[0][1] = 2.0f;

    deconv.octantFourierShift(data);

    EXPECT_TRUE(approxEqualComplex(data[0], 42.0f, 7.0f, 1e-3f))
        << "Center spike should move to origin";
    EXPECT_TRUE(approxEqualComplex(data[centerIdx], 1.0f, 2.0f, 1e-3f))
        << "Origin value should move to center";
}

TEST_F(CPUComputeBackendTest, OctantShiftRealMovesCenterToOrigin) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{8, 8, 8};

    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) data[i] = 0.0f;
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    data[centerIdx] = 42.0f;
    data[0] = 1.0f;

    deconv.octantFourierShift(data);

    EXPECT_TRUE(approxEqual(data[0], 42.0f, 1e-3f)) << "Center spike should move to origin";
    EXPECT_TRUE(approxEqual(data[centerIdx], 1.0f, 1e-3f)) << "Origin value should move to center";
}

TEST_F(CPUComputeBackendTest, OctantShiftMovesCenterToOriginOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = 0.0f;
        data[i][1] = 0.0f;
    }
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    data[centerIdx][0] = 42.0f;
    data[centerIdx][1] = 7.0f;

    deconv.octantFourierShift(data);

    EXPECT_TRUE(approxEqualComplex(data[0], 42.0f, 7.0f, 1e-3f))
        << "Center spike should move to origin even for odd dimensions";
}

TEST_F(CPUComputeBackendTest, OctantShiftRealMovesCenterToOriginOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) data[i] = 0.0f;
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    data[centerIdx] = 42.0f;

    deconv.octantFourierShift(data);

    EXPECT_TRUE(approxEqual(data[0], 42.0f, 1e-3f))
        << "Center spike should move to origin even for odd dimensions";
}

TEST_F(CPUComputeBackendTest, OctantShiftInvolutionOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i);
        data[i][1] = static_cast<real_t>(i * 2);
    }

    deconv.octantFourierShift(data);
    deconv.octantFourierShift(data);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqualComplex(data[i], static_cast<real_t>(i), static_cast<real_t>(i * 2), 1e-3f))
            << "Double shift must be identity for odd dimensions";
    }
}

// Reference implementation for complex data stored as interleaved real_t
// (2 per element).  complex_t is float[2] which cannot be stored in a
// std::vector on libc++ (macOS), so we use std::vector<real_t> instead.
static void referenceShiftComplex(std::vector<real_t>& buf, size_t W, size_t H, size_t D) {
    size_t halfW = W / 2, halfH = H / 2, halfD = D / 2;
    for (size_t z = 0; z < halfD; ++z)
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x) {
                size_t i1 = z * H * W + y * W + x;
                size_t i2 = ((z + halfD) % D) * H * W +
                            ((y + halfH) % H) * W +
                            ((x + halfW) % W);
                if (i1 != i2) {
                    std::swap(buf[i1 * 2],     buf[i2 * 2]);
                    std::swap(buf[i1 * 2 + 1], buf[i2 * 2 + 1]);
                }
            }
}

TEST_F(CPUComputeBackendTest, OctantShiftMatchesReferenceOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    std::vector<real_t> ref(shape.getVolume() * 2);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i][0] = static_cast<real_t>(i) * 0.3f;
        data[i][1] = static_cast<real_t>(i) * 0.7f;
        ref[i * 2]     = static_cast<real_t>(i) * 0.3f;
        ref[i * 2 + 1] = static_cast<real_t>(i) * 0.7f;
    }

    deconv.octantFourierShift(data);
    referenceShiftComplex(ref, shape.width, shape.height, shape.depth);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqualComplex(data[i], ref[i * 2], ref[i * 2 + 1], 1e-3f))
            << "Shift must match reference at index " << i;
    }
}

TEST_F(CPUComputeBackendTest, OctantShiftPaddedLayout) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{8, 6, 4};

    // FFTW in-place r2c layout: physical width > logical width, with padding.
    RealData data = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    ASSERT_GT(data.getPadding(), 0u) << "Test requires a padded layout";

    std::vector<real_t> ref(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        data[i] = static_cast<real_t>(i) * 0.5f + 1.0f;
        ref[i] = static_cast<real_t>(i) * 0.5f + 1.0f;
    }

    deconv.octantFourierShift(data);
    referenceShift(ref, shape.width, shape.height, shape.depth);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(data[i], ref[i], 1e-3f))
            << "Strided access must match reference at index " << i;
    }

    // Involution must also hold under padding.
    deconv.octantFourierShift(data);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_TRUE(approxEqual(data[i], static_cast<real_t>(i) * 0.5f + 1.0f, 1e-3f))
            << "Double shift must be identity even with padded layout";
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
                EXPECT_TRUE(approxEqual(gradX[z * shape.height * shape.width + y * shape.width + x], 1.0f, 1e-4f));
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
    IBackend& backend2 = getManager().createBackendForCurrentThread(cfg);

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
