#include <iostream>
#include <cmath>
#include <thread>
#include <atomic>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"

static bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
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
    std::cout << "=== Test: Concurrent Use of Different Backends ===" << std::endl;

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
    std::atomic<int> errors{0};

    auto worker = [&](IDeconvolutionBackend& deconv, IBackendMemoryManager& memMgr, float value) {
        try {
            ComplexData input = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData output = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData roundtrip = memMgr.allocateMemoryOnDeviceComplexFull(shape);

            for (int i = 0; i < shape.getVolume(); ++i) {
                input[i][0] = value;
                input[i][1] = 0.0f;
            }

            deconv.forwardFFT(input, output);
            deconv.backwardFFT(output, roundtrip);

            for (int i = 0; i < shape.getVolume(); ++i) {
                if (!approxEqual(roundtrip[i][0], value, 1e-3f) ||
                    !approxEqual(roundtrip[i][1], 0.0f, 1e-3f)) {
                    errors++;
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "  Thread exception: " << e.what() << std::endl;
            errors++;
        }
    };

    // Run two threads, each using a different backend
    std::thread t1(worker, std::ref(deconv1), std::ref(memMgr1), 1.0f);
    std::thread t2(worker, std::ref(deconv2), std::ref(memMgr2), 2.0f);

    t1.join();
    t2.join();

    if (errors.load() != 0) {
        std::cerr << "FAIL: Both backends should work concurrently without errors" << std::endl;
        return 1;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
