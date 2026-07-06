#include <gtest/gtest.h>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"
#include <cmath>

class IndependentFFTTest : public ::testing::Test {
protected:
    CPUBackendManager* mgr = nullptr;

    void SetUp() override {
        static CPUBackendManager manager;
        static bool initialized = false;
        if (!initialized) {
            manager.init([](const std::string& context, const std::string& msg, LogLevel level) {
                if (level >= LogLevel::ERROR) {
                    std::cerr << "[" << context << "] " << msg << std::endl;
                }
            });
            initialized = true;
        }
        mgr = &manager;
    }
};

TEST_F(IndependentFFTTest, TwoBackendsIndependentFFTRoundTrip) {
    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend1 = mgr->createBackendForCurrentThread(config);
    IBackend& backend2 = mgr->createBackendForCurrentThread(config);

    IComputeBackend& compute1 = backend1.mutableComputeManager();
    IComputeBackend& compute2 = backend2.mutableComputeManager();
    IBackendMemoryManager& memMgr1 = backend1.mutableMemoryManager();
    IBackendMemoryManager& memMgr2 = backend2.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);

    ComplexData in1 = memMgr1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData out1 = memMgr1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData rt1 = memMgr1.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        in1.access(i)[0] = 0.0f;
        in1.access(i)[1] = 0.0f;
    }
    in1.access(0)[0] = 1.0f;

    ComplexData in2 = memMgr2.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData out2 = memMgr2.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData rt2 = memMgr2.allocateMemoryOnDeviceComplexFull(shape);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        in2.access(i)[0] = 2.0f;
        in2.access(i)[1] = 0.0f;
    }

    compute1.forwardFFT(in1, out1);
    compute1.backwardFFT(out1, rt1);

    compute2.forwardFFT(in2, out2);
    compute2.backwardFFT(out2, rt2);

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(rt1.access(i)[0], in1.access(i)[0], 1e-3f)
            << "Backend1 FFT round-trip should recover impulse at index " << i;
        EXPECT_NEAR(rt1.access(i)[1], in1.access(i)[1], 1e-3f)
            << "Backend1 FFT round-trip should recover impulse at index " << i;
        EXPECT_NEAR(rt2.access(i)[0], in2.access(i)[0], 1e-3f)
            << "Backend2 FFT round-trip should recover constant at index " << i;
        EXPECT_NEAR(rt2.access(i)[1], in2.access(i)[1], 1e-3f)
            << "Backend2 FFT round-trip should recover constant at index " << i;
    }
}
