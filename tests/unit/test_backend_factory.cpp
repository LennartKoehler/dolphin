#include <gtest/gtest.h>
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphin/Logging.h"

class BackendFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(BackendFactoryTest, GetCPUBackendManager) {
    auto& factory = BackendFactory::getInstance();
    auto& manager = factory.getBackendManager("cpu");
    EXPECT_EQ(manager.getNumberDevices(), 1);
}

TEST_F(BackendFactoryTest, GetInvalidBackendFallsBackToCPU) {
    auto& factory = BackendFactory::getInstance();
    auto& manager = factory.getBackendManager("nonexistent_backend");
    EXPECT_EQ(manager.getNumberDevices(), 1);
}

TEST_F(BackendFactoryTest, GetIBackend) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    EXPECT_EQ(backend.getDeviceString(), "cpu");
}

TEST_F(BackendFactoryTest, GetMemoryManager) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& memMgr = factory.getBackend<IBackendMemoryManager>(config);
    EXPECT_NO_THROW(memMgr.getAvailableMemory());
}

TEST_F(BackendFactoryTest, GetDefaultBackendMemoryManager) {
    auto& factory = BackendFactory::getInstance();
    auto& memMgr = factory.getDefaultBackendMemoryManager();
    EXPECT_NO_THROW(memMgr.getAvailableMemory());
}

TEST_F(BackendFactoryTest, CPUMemoryAllocation) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& memMgr = factory.getBackend<IBackendMemoryManager>(config);

    CuboidShape shape{8, 8, 8};
    RealData data = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    EXPECT_TRUE(data.isValid());
    EXPECT_EQ(data.getSize().width, 8);
    EXPECT_EQ(data.getSize().height, 8);
    EXPECT_EQ(data.getSize().depth, 8);
}

TEST_F(BackendFactoryTest, CPUComplexAllocation) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& memMgr = factory.getBackend<IBackendMemoryManager>(config);

    CuboidShape shape{4, 4, 4};
    ComplexData data = memMgr.allocateMemoryOnDeviceComplex(shape);
    EXPECT_TRUE(data.isValid());
}

TEST_F(BackendFactoryTest, CPUFFTRoundTrip) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    CuboidShape shape{8, 8, 8};
    RealData realData = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    ComplexData complexData = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        realData.access(i) = 1.0f;
    }

    compute.forwardFFT(realData, complexData);
    compute.backwardFFT(complexData, realData);

    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        EXPECT_NEAR(realData.access(i), 1.0f, 0.01f);
    }
}

TEST_F(BackendFactoryTest, CPUComplexMultiplication) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    CuboidShape shape{4, 4, 4};
    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        a.access(i)[0] = 1.0f; a.access(i)[1] = 0.0f;
        b.access(i)[0] = 2.0f; b.access(i)[1] = 0.0f;
    }

    compute.complexMultiplication(a, b, result);

    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        EXPECT_NEAR(result.access(i)[0], 2.0f, 0.001f);
        EXPECT_NEAR(result.access(i)[1], 0.0f, 0.001f);
    }
}

TEST_F(BackendFactoryTest, CPUOctantFourierShiftRoundTrip) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& compute = backend.getComputeManager();
    auto& memMgr = backend.getMemoryManager();

    CuboidShape shape{8, 8, 8};
    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<float> original(shape.getVolume() * 2);
    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        data.access(i)[0] = static_cast<float>(i);
        data.access(i)[1] = static_cast<float>(i) * 0.5f;
        original[i * 2] = static_cast<float>(i);
        original[i * 2 + 1] = static_cast<float>(i) * 0.5f;
    }

    compute.octantFourierShift(data);
    compute.octantFourierShift(data);

    for (size_t i = 0; i < static_cast<size_t>(shape.getVolume()); i++) {
        EXPECT_NEAR(data.access(i)[0], original[i * 2], 0.01f);
        EXPECT_NEAR(data.access(i)[1], original[i * 2 + 1], 0.01f);
    }
}

TEST_F(BackendFactoryTest, CPUBackendOwnership) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);

    bool hasMemory = backend.hasMemoryManager();
    EXPECT_TRUE(hasMemory);
}

TEST_F(BackendFactoryTest, CPUMemoryTracking) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& memMgr = factory.getBackend<IBackendMemoryManager>(config);

    size_t before = memMgr.getAllocatedMemory();
    {
        CuboidShape shape{16, 16, 16};
        ComplexData data = memMgr.allocateMemoryOnDeviceComplex(shape);
        size_t after = memMgr.getAllocatedMemory();
        EXPECT_GT(after, before);
    }
}

TEST_F(BackendFactoryTest, CPUSync) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    EXPECT_NO_THROW(backend.sync());
}

TEST_F(BackendFactoryTest, MultipleThreadsBackend) {
    auto& factory = BackendFactory::getInstance();
    BackendConfig config{4, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    EXPECT_EQ(backend.getDeviceString(), "cpu");
}

TEST_F(BackendFactoryTest, BackendManagerClone) {
    auto& factory = BackendFactory::getInstance();
    auto& manager = factory.getBackendManager("cpu");
    BackendConfig config{1, "cpu"};
    auto& backend = factory.getBackend<IBackend>(config);
    auto& cloned = manager.cloneSharedMemory(backend, config);
    EXPECT_EQ(cloned.getDeviceString(), "cpu");
}
