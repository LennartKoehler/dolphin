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
#include "cuda_backend/CUDABackend.h"
#include "cuda_backend/CUDABackendManager.h"

static bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool approxEqualComplex(const complex_t& a, float real, float imag, float eps = 1e-4f) {
    return approxEqual(a[0], real, eps) && approxEqual(a[1], imag, eps);
}

class CUDABackendTest : public ::testing::Test {
protected:
    static CUDABackendManager& getManager() {
        static CUDABackendManager manager;
        static bool initialized = false;
        if (!initialized) {
            manager.init([](const std::string& context, const std::string& msg, LogLevel level) {
                if (level >= LogLevel::ERROR) {
                    std::cerr << "[" << context << "] " << msg << std::endl;
                }
            });
            initialized = true;
        }
        return manager;
    }

    BackendConfig config{nThreads: 1, backendName: "cuda_test"};
    IBackend* backend = nullptr;

    void SetUp() override {
        backend = &getManager().createBackendForCurrentThread(config);
        ASSERT_NE(backend, nullptr);
    }

    std::vector<real_t> readbackReal(const RealData& deviceData) {
        size_t volume = deviceData.getSize().getVolume();
        size_t width = deviceData.getSize().width;
        size_t padding = deviceData.getPadding();
        std::vector<real_t> host(volume);

        if (padding > 0) {
            size_t srcPitch = (width + padding) * sizeof(real_t);
            size_t dstPitch = width * sizeof(real_t);
            cudaMemcpy2DAsync(host.data(), dstPitch,
                              deviceData.getData(), srcPitch,
                              dstPitch, deviceData.getSize().height * deviceData.getSize().depth,
                              cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyAsync(host.data(), deviceData.getData(),
                            volume * sizeof(real_t), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        return host;
    }

    std::vector<float> readbackComplexMag(const ComplexData& deviceData) {
        IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
        size_t bytes = deviceData.getDataBytes();
        std::vector<complex_t> host(deviceData.getSize().getVolume());
        cudaMemcpyAsync(host.data(), deviceData.getData(), bytes, cudaMemcpyDeviceToHost);
        std::vector<float> mags(host.size());
        for (size_t i = 0; i < host.size(); ++i) {
            mags[i] = std::sqrt(host[i][0] * host[i][0] + host[i][1] * host[i][1]);
        }
        return mags;
    }

    void writeRealToDevice(RealData& deviceData, std::vector<real_t>& hostData) {
        size_t width = deviceData.getSize().width;
        size_t padding = deviceData.getPadding();

        if (padding > 0) {
            size_t dstPitch = (width + padding) * sizeof(real_t);
            size_t srcPitch = width * sizeof(real_t);
            cudaMemcpy2DAsync(deviceData.getData(), dstPitch,
                              hostData.data(), srcPitch,
                              srcPitch, deviceData.getSize().height * deviceData.getSize().depth,
                              cudaMemcpyHostToDevice);
        } else {
            cudaMemcpyAsync(deviceData.getData(), hostData.data(),
                            deviceData.getSize().getVolume() * sizeof(real_t),
                            cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
    }

    void writeComplexToDevice(ComplexData& deviceData, std::vector<complex_t>& hostData) {
        IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
        memMgr.memCopy(hostData.data(), deviceData.getData(),
                       deviceData.getSize().getVolume() * sizeof(complex_t), deviceData.getSize());
    }
};

class CUDAMemoryManagerTest : public CUDABackendTest {};
class CUDAComputeBackendTest : public CUDABackendTest {};
class CUDABackendManagerTest : public CUDABackendTest {};
class CUDAOwnershipTest : public CUDABackendTest {};

// ============================================================================
// BackendManager tests
// ============================================================================

TEST_F(CUDABackendManagerTest, ManagerInit) {
    GTEST_LOG_(INFO) << "Detected number of cuda devices" << getManager().getNumberDevices();
}

TEST_F(CUDABackendManagerTest, GetBackendReturnsValidBackend) {
    IBackend& b = getManager().createBackendForCurrentThread(config);
    EXPECT_NE(b.getDeviceString().find("cuda"), std::string::npos);
    EXPECT_TRUE(b.hasMemoryManager());
    EXPECT_TRUE(b.ownsComputeBackend());
    EXPECT_TRUE(b.ownsMemoryManager());
    EXPECT_NE(b.getMemoryManagerPtr(), nullptr);
}

// TEST_F(CUDABackendManagerTest, GetComputeBackendReturnsValidCompute) {
//     IComputeBackend& compute = getManager().getComputeBackend(config);
//     EXPECT_NE(compute.getDeviceString().find("cuda"), std::string::npos);
// }
//
// TEST_F(CUDABackendManagerTest, GetBackendMemoryManagerReturnsValidManager) {
//     IBackendMemoryManager& mem = getManager().getBackendMemoryManager(config);
//     EXPECT_NE(mem.getDeviceString().find("cuda"), std::string::npos);
// }

TEST_F(CUDABackendManagerTest, MultipleBackendsAreDistinct) {
    BackendConfig config1{1, "backend1"};
    BackendConfig config2{1, "backend2"};

    IBackend& b1 = getManager().createBackendForCurrentThread(config1);
    IBackend& b2 = getManager().createBackendForCurrentThread(config2);

    EXPECT_NE(&b1, &b2);
    EXPECT_NE(b1.getMemoryManagerPtr(), b2.getMemoryManagerPtr());
    EXPECT_TRUE(b1.ownsComputeBackend());
    EXPECT_TRUE(b1.ownsMemoryManager());
    EXPECT_TRUE(b2.ownsComputeBackend());
    EXPECT_TRUE(b2.ownsMemoryManager());
}

// TEST_F(CUDABackendManagerTest, CloneReturnsBackend) {
//     IBackend& original = getManager().getBackend(config);
//     IBackend& cloned = getManager().clone(original, config);
//     EXPECT_NE(&cloned, &original);
//     EXPECT_NE(cloned.getDeviceString().find("cuda"), std::string::npos);
// }

TEST_F(CUDABackendManagerTest, CloneSharedMemoryReturnsSeparateBackend) {
    IBackend& original = getManager().createBackendForCurrentThread(config);
    IBackend& shared = getManager().createBackendSharedMemoryForCurrentThread(original, config);
    EXPECT_NE(&original, &shared);
    EXPECT_TRUE(shared.hasMemoryManager());
    EXPECT_TRUE(shared.ownsComputeBackend());
    EXPECT_TRUE(shared.ownsMemoryManager());
}

TEST_F(CUDABackendManagerTest, SetThreadDistribution) {
    size_t ioThreads = 0, workerThreads = 0;
    BackendConfig ioConfig, workerConfig;
    getManager().setThreadDistribution(4, ioThreads, workerThreads, ioConfig, workerConfig);
}


// ============================================================================
// BackendMemoryManager tests
// ============================================================================

TEST_F(CUDAMemoryManagerTest, AllocateReal) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
    EXPECT_EQ(data.getSize(), shape);
}

TEST_F(CUDAMemoryManagerTest, AllocateComplex) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplex(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
}

TEST_F(CUDAMemoryManagerTest, AllocateRealFFTInPlace) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    RealData data = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
}

TEST_F(CUDAMemoryManagerTest, AllocateComplexFull) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    EXPECT_NE(data.getData(), nullptr);
    EXPECT_TRUE(data.isValid());
}

TEST_F(CUDAMemoryManagerTest, MemoryTrackingIncreasesOnAlloc) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    size_t before = memMgr.getAllocatedMemory();
    {
        CuboidShape shape(32, 32, 16);
        RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
        EXPECT_GT(memMgr.getAllocatedMemory(), before);
    }
    EXPECT_EQ(memMgr.getAllocatedMemory(), before);
}

TEST_F(CUDAMemoryManagerTest, MemoryCopy) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);

    std::vector<real_t> hostData(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostData[i] = static_cast<real_t>(i * 3);

    void* devicePtr = memMgr.copyDataToDevice(hostData.data(), bytes, shape);
    ASSERT_NE(devicePtr, nullptr);

    RealData dst = memMgr.allocateMemoryOnDeviceReal(shape);
    memMgr.memCopy(devicePtr, dst.getData(), bytes, shape);

    auto readback = readbackReal(dst);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_FLOAT_EQ(readback[i], hostData[i]);

    memMgr.freeMemoryOnDevice(devicePtr, bytes);
}

TEST_F(CUDAMemoryManagerTest, CopyDataToDevice) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);

    std::vector<real_t> hostData(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostData[i] = static_cast<real_t>(i * 2);

    void* devicePtr = memMgr.copyDataToDevice(hostData.data(), bytes, shape);
    EXPECT_NE(devicePtr, nullptr);

    std::vector<real_t> readback(shape.getVolume());
    memMgr.memCopy(devicePtr, readback.data(), bytes, shape);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_FLOAT_EQ(readback[i], hostData[i]);

    memMgr.freeMemoryOnDevice(devicePtr, bytes);
}

TEST_F(CUDAMemoryManagerTest, MoveDataBetweenBackends) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);

    std::vector<real_t> hostData(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostData[i] = static_cast<real_t>(i * 5);

    void* devicePtr = memMgr.copyDataToDevice(hostData.data(), bytes, shape);
    ASSERT_NE(devicePtr, nullptr);

    void* movedPtr = memMgr.moveDataFromDevice(devicePtr, bytes, shape, memMgr);
    EXPECT_NE(movedPtr, nullptr);

    std::vector<real_t> readback(shape.getVolume());
    memMgr.memCopy(movedPtr, readback.data(), bytes, shape);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_FLOAT_EQ(readback[i], hostData[i]);

    memMgr.freeMemoryOnDevice(movedPtr, bytes);
}

TEST_F(CUDAMemoryManagerTest, ReinterpretRealToComplex) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 8, 4);
    RealData realData = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    EXPECT_NE(realData.getData(), nullptr);

    DataView<complex_t> complexView = memMgr.reinterpret(realData);
    EXPECT_NE(complexView.getData(), nullptr);
    EXPECT_TRUE(complexView.isValid());
    EXPECT_EQ(complexView.getSize().width, shape.width / 2 + 1);
}

TEST_F(CUDAMemoryManagerTest, ReinterpretComplexToReal) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 8, 4);
    RealData realData = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    DataView<complex_t> complexView = memMgr.reinterpret(realData);
    DataView<real_t> realView = memMgr.reinterpret(complexView);
    EXPECT_NE(realView.getData(), nullptr);
    EXPECT_TRUE(realView.isValid());
}

TEST_F(CUDAMemoryManagerTest, IsOnDevice) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    EXPECT_TRUE(memMgr.isOnDevice(data.getData()));
    EXPECT_FALSE(memMgr.isOnDevice(nullptr));
}

TEST_F(CUDAMemoryManagerTest, GetAllocatedMemory) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_NO_THROW({
        size_t mem = memMgr.getAllocatedMemory();
        (void)mem;
    });
}

TEST_F(CUDAMemoryManagerTest, GetAvailableMemory) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_NO_THROW({
        size_t mem = memMgr.getAvailableMemory();
        (void)mem;
    });
}

TEST_F(CUDAMemoryManagerTest, FreeMemoryOnDevice) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    void* ptr = memMgr.allocateMemoryOnDevice(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(memMgr.freeMemoryOnDevice(ptr, 1024));
}

TEST_F(CUDAMemoryManagerTest, SetMemoryLimit) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_NO_THROW(memMgr.setMemoryLimit(1024 * 1024 * 100));
}

TEST_F(CUDAMemoryManagerTest, CreateCopy) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);
    size_t bytes = shape.getVolume() * sizeof(real_t);

    std::vector<real_t> hostData(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostData[i] = static_cast<real_t>(i * 7);

    void* devicePtr = memMgr.copyDataToDevice(hostData.data(), bytes, shape);
    RealData src(&memMgr, static_cast<real_t*>(devicePtr), shape, shape, bytes, 0);

    RealData copy = memMgr.createCopy(src);
    EXPECT_NE(copy.getData(), nullptr);
    EXPECT_TRUE(copy.isValid());

    auto readback = readbackReal(copy);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_FLOAT_EQ(readback[i], hostData[i]);

    src.setData(nullptr);
    memMgr.freeMemoryOnDevice(devicePtr, bytes);
}

TEST_F(CUDAMemoryManagerTest, Sync) {
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    EXPECT_NO_THROW(memMgr.sync());
}

// ============================================================================
// ComputeBackend tests
// ============================================================================

TEST_F(CUDAComputeBackendTest, ComplexFFTRoundTrip) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostIn[i][0] = 0.0f;
        hostIn[i][1] = 0.0f;
    }
    hostIn[0][0] = 1.0f;
    writeComplexToDevice(input, hostIn);

    deconv.forwardFFT(input, output);
    deconv.backwardFFT(output, roundtrip);
    backend->sync();

    std::vector<complex_t> hostRt(roundtrip.getSize().getVolume());
    memMgr.memCopy(roundtrip.getData(), hostRt.data(),
                   roundtrip.getSize().getVolume() * sizeof(complex_t), roundtrip.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(hostRt[i][0], hostIn[i][0], 1e-2f);
        EXPECT_NEAR(hostRt[i][1], hostIn[i][1], 1e-2f);
    }
}

TEST_F(CUDAComputeBackendTest, RealFFTRoundTrip) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    // ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    DataView<complex_t> complexOut = realIn.reinterpret();
    // RealData realOut = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);

    std::vector<real_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostIn[i] = static_cast<real_t>(i % 10) * 0.1f;

    writeRealToDevice(realIn, hostIn);

    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realIn);
    backend->sync();

    auto readback = readbackReal(realIn);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_NEAR(readback[i], hostIn[i], 1e-2f);
}

TEST_F(CUDAComputeBackendTest, ComplexMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostA(shape.getVolume()), hostB(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostA[i][0] = static_cast<real_t>(i + 1);
        hostA[i][1] = static_cast<real_t>(i + 2);
        hostB[i][0] = static_cast<real_t>(i + 3);
        hostB[i][1] = static_cast<real_t>(i + 4);
    }
    writeComplexToDevice(a, hostA);
    writeComplexToDevice(b, hostB);

    deconv.complexMultiplication(a, b, result);
    backend->sync();

    std::vector<complex_t> hostRt(result.getSize().getVolume());
    memMgr.memCopy(result.getData(), hostRt.data(),
                   result.getSize().getVolume() * sizeof(complex_t), result.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = hostA[i][0], ia = hostA[i][1];
        real_t rb = hostB[i][0], ib = hostB[i][1];
        EXPECT_NEAR(hostRt[i][0], ra*rb - ia*ib, 1e-2f);
        EXPECT_NEAR(hostRt[i][1], ra*ib + ia*rb, 1e-2f);
    }
}

TEST_F(CUDAComputeBackendTest, ComplexAddition) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostA(shape.getVolume()), hostB(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostA[i][0] = static_cast<real_t>(i + 1);
        hostA[i][1] = static_cast<real_t>(i + 2);
        hostB[i][0] = static_cast<real_t>(i + 3);
        hostB[i][1] = static_cast<real_t>(i + 4);
    }
    writeComplexToDevice(a, hostA);
    writeComplexToDevice(b, hostB);

    deconv.complexAddition(a, b, result);
    backend->sync();

    std::vector<complex_t> hostRt(result.getSize().getVolume());
    memMgr.memCopy(result.getData(), hostRt.data(),
                   result.getSize().getVolume() * sizeof(complex_t), result.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(hostRt[i][0], hostA[i][0] + hostB[i][0], 1e-3f);
        EXPECT_NEAR(hostRt[i][1], hostA[i][1] + hostB[i][1], 1e-3f);
    }
}

TEST_F(CUDAComputeBackendTest, RealMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData a = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData b = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData result = memMgr.allocateMemoryOnDeviceReal(shape);

    std::vector<real_t> hostA(shape.getVolume()), hostB(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostA[i] = static_cast<real_t>(i + 1);
        hostB[i] = static_cast<real_t>(i + 2);
    }
    writeRealToDevice(a, hostA);
    writeRealToDevice(b, hostB);

    deconv.multiplication(a, b, result);
    backend->sync();

    auto readback = readbackReal(result);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_NEAR(readback[i], hostA[i] * hostB[i], 1e-2f);
}

TEST_F(CUDAComputeBackendTest, RealScalarMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData a = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData result = memMgr.allocateMemoryOnDeviceReal(shape);

    std::vector<real_t> hostA(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostA[i] = static_cast<real_t>(i + 1);
    writeRealToDevice(a, hostA);

    real_t scalar = 2.5f;
    deconv.scalarMultiplication(a, scalar, result);
    backend->sync();

    auto readback = readbackReal(result);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_NEAR(readback[i], hostA[i] * scalar, 1e-2f);
}

TEST_F(CUDAComputeBackendTest, ComplexScalarMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostA(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostA[i][0] = static_cast<real_t>(i + 1);
        hostA[i][1] = static_cast<real_t>(i + 2);
    }
    writeComplexToDevice(a, hostA);

    complex_t scalar = {2.0f, 3.0f};
    deconv.scalarMultiplication(a, scalar, result);
    backend->sync();

    std::vector<complex_t> hostRt(result.getSize().getVolume());
    memMgr.memCopy(result.getData(), hostRt.data(),
                   result.getSize().getVolume() * sizeof(complex_t), result.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = hostA[i][0], ia = hostA[i][1];
        EXPECT_NEAR(hostRt[i][0], ra*scalar[0] - ia*scalar[1], 1e-2f);
        EXPECT_NEAR(hostRt[i][1], ra*scalar[1] + ia*scalar[0], 1e-2f);
    }
}

TEST_F(CUDAComputeBackendTest, ComplexMultiplicationWithConjugate) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostA(shape.getVolume()), hostB(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostA[i][0] = static_cast<real_t>(i + 1);
        hostA[i][1] = static_cast<real_t>(i + 2);
        hostB[i][0] = static_cast<real_t>(i + 3);
        hostB[i][1] = static_cast<real_t>(i + 4);
    }
    writeComplexToDevice(a, hostA);
    writeComplexToDevice(b, hostB);

    deconv.complexMultiplicationWithConjugate(a, b, result);
    backend->sync();

    std::vector<complex_t> hostRt(result.getSize().getVolume());
    memMgr.memCopy(result.getData(), hostRt.data(),
                   result.getSize().getVolume() * sizeof(complex_t), result.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        real_t ra = hostA[i][0], ia = hostA[i][1];
        real_t rb = hostB[i][0], ib = -hostB[i][1];
        EXPECT_NEAR(hostRt[i][0], ra*rb - ia*ib, 1e-2f);
        EXPECT_NEAR(hostRt[i][1], ra*ib + ia*rb, 1e-2f);
    }
}

TEST_F(CUDAComputeBackendTest, OctantFourierShiftDoubleIsIdentity) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 8);

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostIn[i][0] = static_cast<real_t>(i);
        hostIn[i][1] = static_cast<real_t>(i * 2);
    }
    writeComplexToDevice(data, hostIn);

    deconv.octantFourierShift(data);
    deconv.octantFourierShift(data);
    backend->sync();

    std::vector<complex_t> hostRt(data.getSize().getVolume());
    memMgr.memCopy(data.getData(), hostRt.data(),
                   data.getSize().getVolume() * sizeof(complex_t), data.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(hostRt[i][0], hostIn[i][0], 1e-2f);
        EXPECT_NEAR(hostRt[i][1], hostIn[i][1], 1e-2f);
    }
}

TEST_F(CUDAComputeBackendTest, TVNormalization) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    RealData gx = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gy = memMgr.allocateMemoryOnDeviceReal(shape);
    RealData gz = memMgr.allocateMemoryOnDeviceReal(shape);

    std::vector<real_t> ones(shape.getVolume(), 1.0f);
    writeRealToDevice(gx, ones);
    writeRealToDevice(gy, ones);
    writeRealToDevice(gz, ones);

    deconv.normalizeTV(gx, gy, gz, 1e-6f);
    backend->sync();

    float expected = 1.0f / std::sqrt(3.0f);
    auto rbX = readbackReal(gx);
    auto rbY = readbackReal(gy);
    auto rbZ = readbackReal(gz);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(rbX[i], expected, 1e-2f);
        EXPECT_NEAR(rbY[i], expected, 1e-2f);
        EXPECT_NEAR(rbZ[i], expected, 1e-2f);
    }
}

TEST_F(CUDAComputeBackendTest, HasNANDoesNotThrow) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<complex_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostIn[i][0] = static_cast<real_t>(i);
        hostIn[i][1] = 0.0f;
    }
    writeComplexToDevice(data, hostIn);
    EXPECT_NO_THROW(deconv.hasNAN(data));
}

// ============================================================================
// Ownership tests
// ============================================================================

TEST_F(CUDAOwnershipTest, TransferComputeBackend) {
    EXPECT_TRUE(backend->ownsComputeBackend());
    auto deconvPtr = backend->releaseComputeBackend();
    EXPECT_NE(deconvPtr, nullptr);
    EXPECT_FALSE(backend->ownsComputeBackend());
    backend->takeOwnership(std::move(deconvPtr));
    EXPECT_TRUE(backend->ownsComputeBackend());
}

TEST_F(CUDAOwnershipTest, TransferMemoryManager) {
    EXPECT_TRUE(backend->ownsMemoryManager());
    auto memPtr = backend->releaseMemoryManager();
    EXPECT_NE(memPtr, nullptr);
    EXPECT_FALSE(backend->ownsMemoryManager());
    backend->takeOwnership(std::move(memPtr));
    EXPECT_TRUE(backend->ownsMemoryManager());
}

TEST_F(CUDAOwnershipTest, ReleaseWithoutOwnershipThrows) {
    auto deconvPtr = backend->releaseComputeBackend();
    EXPECT_THROW(backend->releaseComputeBackend(), std::runtime_error);
    backend->takeOwnership(std::move(deconvPtr));
}

TEST_F(CUDAOwnershipTest, SharedMemoryManager) {
    auto sharedMem = backend->getSharedMemoryManager();
    EXPECT_NE(sharedMem, nullptr);
    EXPECT_EQ(sharedMem.get(), backend->getMemoryManagerPtr());
}

