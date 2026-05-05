#include <iostream>
#include <cmath>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"

static bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool approxEqualComplex(const complex_t& a, float real, float imag, float eps = 1e-4f) {
    return approxEqual(a[0], real, eps) && approxEqual(a[1], imag, eps);
}

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

int main() {
    std::cout << "=== Test: Independent FFT on Different Backends ===" << std::endl;

    CPUBackendManager& mgr = getManager();

    BackendConfig config;
    config.nThreads = 1;

    IBackend& backend1 = mgr.getBackend(config);
    IBackend& backend2 = mgr.getBackend(config);

    IDeconvolutionBackend& deconv1 = backend1.mutableDeconvManager();
    IDeconvolutionBackend& deconv2 = backend2.mutableDeconvManager();
    IBackendMemoryManager& memMgr1 = backend1.mutableMemoryManager();
    IBackendMemoryManager& memMgr2 = backend2.mutableMemoryManager();

    CuboidShape shape(16, 16, 8);

    // Backend1: impulse at origin
    ComplexData in1 = memMgr1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData out1 = memMgr1.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData rt1 = memMgr1.allocateMemoryOnDeviceComplexFull(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        in1[i][0] = 0.0f;
        in1[i][1] = 0.0f;
    }
    in1[0][0] = 1.0f;

    // Backend2: constant input
    ComplexData in2 = memMgr2.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData out2 = memMgr2.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData rt2 = memMgr2.allocateMemoryOnDeviceComplexFull(shape);

    for (int i = 0; i < shape.getVolume(); ++i) {
        in2[i][0] = 2.0f;
        in2[i][1] = 0.0f;
    }

    deconv1.forwardFFT(in1, out1);
    deconv1.backwardFFT(out1, rt1);

    deconv2.forwardFFT(in2, out2);
    deconv2.backwardFFT(out2, rt2);

    // Verify round-trip independently
    for (int i = 0; i < shape.getVolume(); ++i) {
        if (!approxEqualComplex(rt1[i], in1[i][0], in1[i][1], 1e-3f)) {
            std::cerr << "FAIL: Backend1 FFT round-trip should recover impulse" << std::endl;
            return 1;
        }
        if (!approxEqualComplex(rt2[i], in2[i][0], in2[i][1], 1e-3f)) {
            std::cerr << "FAIL: Backend2 FFT round-trip should recover constant" << std::endl;
            return 1;
        }
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
