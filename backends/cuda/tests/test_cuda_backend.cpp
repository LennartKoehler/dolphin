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

#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include "dolphinbackend/IComputeBackend.h"
#include "cuda_backend/CUDABackend.h"
#include "cuda_backend/CUDABackendManager.h"
#include "dolphinbackend/Exceptions.h"

struct ComplexHost {
    real_t data[2];
    real_t& operator[](size_t i) { return data[i]; }
    const real_t& operator[](size_t i) const { return data[i]; }
};
static_assert(sizeof(ComplexHost) == sizeof(complex_t), "layout mismatch");

static bool approxEqual(real_t a, real_t b, real_t eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool approxEqualComplex(const ComplexHost& a, real_t real, real_t imag, real_t eps = 1e-4f) {
    return approxEqual(a[0], real, eps) && approxEqual(a[1], imag, eps);
}

// Reference values for 4x4x4 Fourier transforms, computed independently (direct DFT).
// Linear index i = d*16 + h*4 + w (width fastest). Forward unnormalized, backward /64.
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

class CUDABackendTest : public ::testing::Test {
protected:
    static CUDABackendManager& getManager() {
        static CUDABackendManager manager;
        static bool initialized = false;
        if (!initialized) {
            manager.init([](const std::string& context, const std::string& msg, LogLevel level) {
                if (level >= LogLevel::LOG_ERROR) {
                    std::cerr << "[" << context << "] " << msg << std::endl;
                }
            });
            initialized = true;
        }
        return manager;
    }

    BackendConfig config{1, "cuda_test"};
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

    std::vector<real_t> readbackComplexMag(const ComplexData& deviceData) {
        IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
        size_t bytes = deviceData.getDataBytes();
        std::vector<ComplexHost> host(deviceData.getSize().getVolume());
        cudaMemcpyAsync(host.data(), deviceData.getData(), bytes, cudaMemcpyDeviceToHost);
        std::vector<real_t> mags(host.size());
        for (size_t i = 0; i < host.size(); ++i) {
            mags[i] = std::sqrt(host[i][0] * host[i][0] + host[i][1] * host[i][1]);
        }
        return mags;
    }

    std::vector<ComplexHost> readbackComplex(const ComplexData& deviceData) {
        size_t volume = deviceData.getSize().getVolume();
        size_t width = deviceData.getSize().width;
        size_t padding = deviceData.getPadding();
        std::vector<ComplexHost> host(volume);

        if (padding > 0) {
            size_t srcPitch = (width + padding) * sizeof(complex_t);
            size_t dstPitch = width * sizeof(complex_t);
            cudaMemcpy2DAsync(host.data(), dstPitch,
                              deviceData.getData(), srcPitch,
                              dstPitch, deviceData.getSize().height * deviceData.getSize().depth,
                              cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyAsync(host.data(), deviceData.getData(),
                            volume * sizeof(complex_t), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        return host;
    }

    // Read a real buffer that has been used for an in-place real FFT: the physical
    // real row is (width + padding) floats = (width/2+1) complex, stored at the row start.
    // Returns (depth*height*(width/2+1)) complex values indexed i = d*(ch*hw) + h*ch + w.
    std::vector<std::pair<float, float>> readbackRealInPlace(RealData& deviceData) {
        size_t width = deviceData.getSize().width;
        size_t height = deviceData.getSize().height;
        size_t depth = deviceData.getSize().depth;
        size_t physReal = width + deviceData.getPadding();
        size_t ch = width / 2 + 1;
        std::vector<real_t> phys(physReal * height * depth);
        cudaMemcpyAsync(phys.data(), deviceData.getData(),
                        phys.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        std::vector<std::pair<float, float>> out(ch * height * depth);
        size_t o = 0;
        for (size_t d = 0; d < depth; ++d)
            for (size_t h = 0; h < height; ++h)
                for (size_t w = 0; w < ch; ++w) {
                    size_t base = (d * height + h) * physReal + 2 * w;
                    out[o++] = { phys[base], phys[base + 1] };
                }
        return out;
    }

    // Write (width/2+1) complex values per row into a real buffer laid out for an
    // in-place real FFT (physical real row = width + padding), zero-padding the rest.
    void writeRealInPlaceComplex(RealData& deviceData, const std::pair<float, float>* c, size_t count) {
        size_t width = deviceData.getSize().width;
        size_t height = deviceData.getSize().height;
        size_t depth = deviceData.getSize().depth;
        size_t physReal = width + deviceData.getPadding();
        size_t ch = width / 2 + 1;
        std::vector<real_t> phys(physReal * height * depth, 0.0f);
        for (size_t d = 0; d < depth; ++d)
            for (size_t h = 0; h < height; ++h)
                for (size_t w = 0; w < ch; ++w) {
                    size_t idx = d * (ch * height) + h * ch + w;
                    size_t base = (d * height + h) * physReal + 2 * w;
                    phys[base] = c[idx].first;
                    phys[base + 1] = c[idx].second;
                }
        (void)count;
        cudaMemcpyAsync(deviceData.getData(), phys.data(),
                        phys.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void writeComplexPadded(ComplexData& deviceData, const std::pair<float, float>* logical, size_t count) {
        size_t width = deviceData.getSize().width;
        size_t height = deviceData.getSize().height;
        size_t depth = deviceData.getSize().depth;
        size_t padding = deviceData.getPadding();

        if (padding > 0) {
            size_t physVol = (width + padding) * height * depth;
            std::vector<ComplexHost> phys(physVol);
            for (size_t i = 0; i < count; ++i) {
                size_t physIdx = i + (i / width) * padding;
                phys[physIdx][0] = logical[i].first;
                phys[physIdx][1] = logical[i].second;
            }
            cudaMemcpyAsync(deviceData.getData(), phys.data(),
                            physVol * sizeof(complex_t), cudaMemcpyHostToDevice);
        } else {
            std::vector<ComplexHost> host(count);
            for (size_t i = 0; i < count; ++i) {
                host[i][0] = logical[i].first;
                host[i][1] = logical[i].second;
            }
            cudaMemcpyAsync(deviceData.getData(), host.data(),
                            host.size() * sizeof(complex_t), cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
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

    void writeComplexToDevice(ComplexData& deviceData, std::vector<ComplexHost>& hostData) {
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

    std::vector<ComplexHost> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostIn[i][0] = 0.0f;
        hostIn[i][1] = 0.0f;
    }
    hostIn[0][0] = 1.0f;
    writeComplexToDevice(input, hostIn);

    deconv.forwardFFT(input, output);
    deconv.backwardFFT(output, roundtrip);
    backend->sync();

    std::vector<ComplexHost> hostRt(roundtrip.getSize().getVolume());
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

TEST_F(CUDAComputeBackendTest, RealFFTRoundTripOutOfPlaceUnpadded) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceReal(shape);
    ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);

    std::vector<real_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostIn[i] = static_cast<real_t>(i % 10) * 0.1f;

    writeRealToDevice(realIn, hostIn);
    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realOut);
    backend->sync();

    auto readback = readbackReal(realOut);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_NEAR(readback[i], hostIn[i], 1e-2f);
}

TEST_F(CUDAComputeBackendTest, RealFFTMixedPadding) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
    ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);

    std::vector<real_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i)
        hostIn[i] = static_cast<real_t>(i % 7) * 0.2f;

    writeRealToDevice(realIn, hostIn);
    deconv.forwardFFT(realIn, complexOut);
    deconv.backwardFFT(complexOut, realOut);
    backend->sync();

    auto readback = readbackReal(realOut);
    for (size_t i = 0; i < shape.getVolume(); ++i)
        EXPECT_NEAR(readback[i], hostIn[i], 1e-2f);
}

TEST_F(CUDAComputeBackendTest, ForwardFFTRejectsFullComplexOutput) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    RealData realIn = memMgr.allocateMemoryOnDeviceReal(shape);
    ComplexData fullOut = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    EXPECT_THROW(deconv.forwardFFT(realIn, fullOut), dolphin::backend::BackendException);
}

TEST_F(CUDAComputeBackendTest, BackwardFFTRejectsFullComplexInput) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(16, 16, 8);

    ComplexData fullIn = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);

    EXPECT_THROW(deconv.backwardFFT(fullIn, realOut), dolphin::backend::BackendException);
}

TEST_F(CUDAComputeBackendTest, RealForwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place: unpadded real input -> complex output
    {
        RealData realIn = memMgr.allocateMemoryOnDeviceReal(shape);
        ComplexData complexOut = memMgr.allocateMemoryOnDeviceComplex(shape);
        std::vector<real_t> hostIn(kRealInput, kRealInput + 64);
        writeRealToDevice(realIn, hostIn);
        deconv.forwardFFT(realIn, complexOut);
        backend->sync();
        auto out = readbackComplex(complexOut);
        for (size_t i = 0; i < 48; ++i) {
            EXPECT_NEAR(out[i][0], kRealFwdRef[i].first, tol) << "OOP fwd real bin " << i;
            EXPECT_NEAR(out[i][1], kRealFwdRef[i].second, tol) << "OOP fwd real bin " << i;
        }
    }
    // In-place: padded real buffer reinterpreted as a complex view
    {
        RealData realIn = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
        DataView<complex_t> complexOut = memMgr.reinterpret(realIn);
        std::vector<real_t> hostIn(kRealInput, kRealInput + 64);
        writeRealToDevice(realIn, hostIn);
        deconv.forwardFFT(realIn, complexOut);
        backend->sync();
        auto out = readbackRealInPlace(realIn);
        for (size_t i = 0; i < 48; ++i) {
            EXPECT_NEAR(out[i].first, kRealFwdRef[i].first, tol) << "IP fwd real bin " << i;
            EXPECT_NEAR(out[i].second, kRealFwdRef[i].second, tol) << "IP fwd real bin " << i;
        }
    }
}

TEST_F(CUDAComputeBackendTest, RealBackwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place: complex input -> unpadded real output
    {
        ComplexData complexIn = memMgr.allocateMemoryOnDeviceComplex(shape);
        RealData realOut = memMgr.allocateMemoryOnDeviceReal(shape);
        writeComplexPadded(complexIn, kRealFwdRef, 48);
        deconv.backwardFFT(complexIn, realOut);
        backend->sync();
        auto out = readbackReal(realOut);
        for (size_t i = 0; i < 64; ++i)
            EXPECT_NEAR(out[i], kRealInput[i], tol) << "OOP bwd real bin " << i;
    }
    // In-place: complex view on a padded real buffer
    {
        RealData realOut = memMgr.allocateMemoryOnDeviceRealFFTInPlace(shape);
        DataView<complex_t> complexIn = memMgr.reinterpret(realOut);
        writeRealInPlaceComplex(realOut, kRealFwdRef, 48);
        deconv.backwardFFT(complexIn, realOut);
        backend->sync();
        auto out = readbackReal(realOut);
        for (size_t i = 0; i < 64; ++i)
            EXPECT_NEAR(out[i], kRealInput[i], tol) << "IP bwd real bin " << i;
    }
}

TEST_F(CUDAComputeBackendTest, ComplexForwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place
    {
        ComplexData in = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        ComplexData out = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        writeComplexPadded(in, kComplexInput, 64);
        deconv.forwardFFT(in, out);
        backend->sync();
        auto rb = readbackComplex(out);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(rb[i][0], kComplexFwdRef[i].first, tol) << "OOP fwd complex bin " << i;
            EXPECT_NEAR(rb[i][1], kComplexFwdRef[i].second, tol) << "OOP fwd complex bin " << i;
        }
    }
    // In-place
    {
        ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        writeComplexPadded(data, kComplexInput, 64);
        deconv.forwardFFT(data, data);
        backend->sync();
        auto rb = readbackComplex(data);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(rb[i][0], kComplexFwdRef[i].first, tol) << "IP fwd complex bin " << i;
            EXPECT_NEAR(rb[i][1], kComplexFwdRef[i].second, tol) << "IP fwd complex bin " << i;
        }
    }
}

TEST_F(CUDAComputeBackendTest, ComplexBackwardFFTMatchesReference) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(4, 4, 4);
    const real_t tol = 1e-3f;

    // Out-of-place
    {
        ComplexData in = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        ComplexData out = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        writeComplexPadded(in, kComplexFwdRef, 64);
        deconv.backwardFFT(in, out);
        backend->sync();
        auto rb = readbackComplex(out);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(rb[i][0], kComplexInput[i].first, tol) << "OOP bwd complex bin " << i;
            EXPECT_NEAR(rb[i][1], kComplexInput[i].second, tol) << "OOP bwd complex bin " << i;
        }
    }
    // In-place
    {
        ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
        writeComplexPadded(data, kComplexFwdRef, 64);
        deconv.backwardFFT(data, data);
        backend->sync();
        auto rb = readbackComplex(data);
        for (size_t i = 0; i < 64; ++i) {
            EXPECT_NEAR(rb[i][0], kComplexInput[i].first, tol) << "IP bwd complex bin " << i;
            EXPECT_NEAR(rb[i][1], kComplexInput[i].second, tol) << "IP bwd complex bin " << i;
        }
    }
}

TEST_F(CUDAComputeBackendTest, ComplexMultiplication) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape(8, 8, 4);

    ComplexData a = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData b = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    ComplexData result = memMgr.allocateMemoryOnDeviceComplexFull(shape);

    std::vector<ComplexHost> hostA(shape.getVolume()), hostB(shape.getVolume());
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

    std::vector<ComplexHost> hostRt(result.getSize().getVolume());
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

    std::vector<ComplexHost> hostA(shape.getVolume()), hostB(shape.getVolume());
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

    std::vector<ComplexHost> hostRt(result.getSize().getVolume());
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

    std::vector<ComplexHost> hostA(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostA[i][0] = static_cast<real_t>(i + 1);
        hostA[i][1] = static_cast<real_t>(i + 2);
    }
    writeComplexToDevice(a, hostA);

    complex_t scalar = {2.0f, 3.0f};
    deconv.scalarMultiplication(a, scalar, result);
    backend->sync();

    std::vector<ComplexHost> hostRt(result.getSize().getVolume());
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

    std::vector<ComplexHost> hostA(shape.getVolume()), hostB(shape.getVolume());
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

    std::vector<ComplexHost> hostRt(result.getSize().getVolume());
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

    std::vector<ComplexHost> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostIn[i][0] = static_cast<real_t>(i);
        hostIn[i][1] = static_cast<real_t>(i * 2);
    }
    writeComplexToDevice(data, hostIn);

    deconv.octantFourierShift(data);
    deconv.octantFourierShift(data);
    backend->sync();

    std::vector<ComplexHost> hostRt(data.getSize().getVolume());
    memMgr.memCopy(data.getData(), hostRt.data(),
                   data.getSize().getVolume() * sizeof(complex_t), data.getSize());

    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(hostRt[i][0], hostIn[i][0], 1e-2f);
        EXPECT_NEAR(hostRt[i][1], hostIn[i][1], 1e-2f);
    }
}

// ============================================================================
// octantFourierShift — regression tests
//
// Semantics: swaps (z,y,x) with ((z+halfD)%D, (y+halfH)%H, (x+halfW)%W) for
// z in [0, halfD).  Involution for all dims; peak at floor(N/2) -> origin.
//
// NOTE: The CUDA complex kernel (global.cu:802) lacks x<width/y<height bounds
// guards and is therefore unsafe for ODD dimensions.  Odd-dim tests here use
// the RealData overload only (its kernel has full bounds checks).  Odd-dim
// ComplexData coverage lives in the CPU test suite.
// ============================================================================

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

TEST_F(CUDAComputeBackendTest, OctantShiftMovesCenterToOrigin) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{8, 8, 8};

    ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
    std::vector<ComplexHost> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        hostIn[i][0] = 0.0f;
        hostIn[i][1] = 0.0f;
    }
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    hostIn[centerIdx][0] = 42.0f;
    hostIn[centerIdx][1] = 7.0f;
    hostIn[0][0] = 1.0f;
    hostIn[0][1] = 2.0f;
    writeComplexToDevice(data, hostIn);

    deconv.octantFourierShift(data);
    backend->sync();

    std::vector<ComplexHost> hostRt(shape.getVolume());
    memMgr.memCopy(data.getData(), hostRt.data(),
                   shape.getVolume() * sizeof(complex_t), shape);

    EXPECT_TRUE(approxEqualComplex(hostRt[0], 42.0f, 7.0f, 1e-2f))
        << "Center spike should move to origin";
    EXPECT_TRUE(approxEqualComplex(hostRt[centerIdx], 1.0f, 2.0f, 1e-2f))
        << "Origin value should move to center";
}

TEST_F(CUDAComputeBackendTest, OctantShiftRealMovesCenterToOrigin) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{8, 8, 8};

    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    std::vector<real_t> hostIn(shape.getVolume(), 0.0f);
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    hostIn[centerIdx] = 42.0f;
    hostIn[0] = 1.0f;
    writeRealToDevice(data, hostIn);

    deconv.octantFourierShift(data);
    backend->sync();

    auto hostRt = readbackReal(data);

    EXPECT_TRUE(approxEqual(hostRt[0], 42.0f, 1e-2f)) << "Center spike should move to origin";
    EXPECT_TRUE(approxEqual(hostRt[centerIdx], 1.0f, 1e-2f)) << "Origin value should move to center";
}

TEST_F(CUDAComputeBackendTest, OctantShiftRealMovesCenterToOriginOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    std::vector<real_t> hostIn(shape.getVolume(), 0.0f);
    size_t cx = shape.width / 2, cy = shape.height / 2, cz = shape.depth / 2;
    size_t centerIdx = cz * shape.height * shape.width + cy * shape.width + cx;
    hostIn[centerIdx] = 42.0f;
    writeRealToDevice(data, hostIn);

    deconv.octantFourierShift(data);
    backend->sync();

    auto hostRt = readbackReal(data);

    EXPECT_TRUE(approxEqual(hostRt[0], 42.0f, 1e-2f))
        << "Center spike should move to origin even for odd dimensions";
}

TEST_F(CUDAComputeBackendTest, OctantShiftRealInvolutionOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    std::vector<real_t> hostIn(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) hostIn[i] = static_cast<real_t>(i);
    writeRealToDevice(data, hostIn);

    deconv.octantFourierShift(data);
    deconv.octantFourierShift(data);
    backend->sync();

    auto hostRt = readbackReal(data);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(hostRt[i], hostIn[i], 1e-2f)
            << "Double shift must be identity for odd dimensions";
    }
}

TEST_F(CUDAComputeBackendTest, OctantShiftRealMatchesReferenceOddDims) {
    IComputeBackend& deconv = backend->mutableComputeManager();
    IBackendMemoryManager& memMgr = backend->mutableMemoryManager();
    CuboidShape shape{5, 7, 9};

    RealData data = memMgr.allocateMemoryOnDeviceReal(shape);
    std::vector<real_t> ref(shape.getVolume());
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        ref[i] = static_cast<real_t>(i) * 0.5f + 1.0f;
    }
    writeRealToDevice(data, ref);

    deconv.octantFourierShift(data);
    backend->sync();

    referenceShift(ref, shape.width, shape.height, shape.depth);
    auto hostRt = readbackReal(data);
    for (size_t i = 0; i < shape.getVolume(); ++i) {
        EXPECT_NEAR(hostRt[i], ref[i], 1e-2f)
            << "Shift must match reference at index " << i;
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

    std::vector<ComplexHost> hostIn(shape.getVolume());
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

