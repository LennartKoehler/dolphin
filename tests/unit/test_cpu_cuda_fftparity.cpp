/*
Temporary diagnostic: CPU vs CUDA in-place real FFT parity.
PoissonMLE is the main user of the IN-PLACE real r2c/c2r FFT path (RL uses
out-of-place). This compares the in-place FFT on both backends with identical
input to see whether a systematic FFT divergence (normalization/layout) exists.
*/
#include <gtest/gtest.h>
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphin/Logging.h"
#include <vector>
#include <cmath>
#include <random>

namespace {
IBackend* getBE(const std::string& name) {
    auto& factory = BackendFactory::getInstance();
    return &factory.getBackend<IBackend>(BackendConfig{1, name});
}

// Fill an in-place real buffer with sig (logical) + 0 padding, forward FFT in place,
// read back the complex spectrum (packed real buffer).
std::vector<real_t> fwdSpectrum(const std::string& name, const CuboidShape& shape,
                                const std::vector<real_t>& sig) {
    IBackend* b = getBE(name);
    const auto& mem = b->getMemoryManager();
    auto& compute = b->mutableComputeManager();
    size_t W = shape.width, H = shape.height, D = shape.depth;
    size_t P = 2 * (W / 2 + 1);

    std::vector<real_t> padded(D * H * P, 0.0f);
    for (size_t z = 0; z < D; ++z)
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x)
                padded[z * H * P + y * P + x] = sig[(z * H + y) * W + x];

    RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(shape);
    mem.memCopy(padded.data(), f.getData(), padded.size() * sizeof(real_t), shape);
    b->sync();

    FFTPlanDescription fReal{PlanDirection::FORWARD, PlanType::REAL, shape, true};
    compute.initializePlan(fReal);

    ComplexView fv = f.reinterpret();
    compute.forwardFFT(f, fv);   // in-place r2c
    b->sync();

    const auto& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();
    RealData h = mem.moveDataFromDevice(f, hostMem);
    size_t n = h.getDataBytes() / sizeof(real_t);
    std::vector<real_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = h.getData()[i];
    return out;
}

// Round trip: forward in-place then backward in-place, read back real.
std::vector<real_t> roundTrip(const std::string& name, const CuboidShape& shape,
                              const std::vector<real_t>& sig) {
    IBackend* b = getBE(name);
    const auto& mem = b->getMemoryManager();
    auto& compute = b->mutableComputeManager();
    size_t W = shape.width, H = shape.height, D = shape.depth;
    size_t P = 2 * (W / 2 + 1);
    std::vector<real_t> padded(D * H * P, 0.0f);
    for (size_t z = 0; z < D; ++z)
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x)
                padded[z * H * P + y * P + x] = sig[(z * H + y) * W + x];

    RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(shape);
    mem.memCopy(padded.data(), f.getData(), padded.size() * sizeof(real_t), shape);
    b->sync();
    FFTPlanDescription fReal{PlanDirection::FORWARD, PlanType::REAL, shape, true};
    FFTPlanDescription bReal{PlanDirection::BACKWARD, PlanType::REAL, shape, true};
    compute.initializePlan(fReal);
    compute.initializePlan(bReal);
    ComplexView fv = f.reinterpret();
    compute.forwardFFT(f, fv);
    compute.backwardFFT(fv, f);   // in-place c2r
    b->sync();
    const auto& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();
    RealData h = mem.moveDataFromDevice(f, hostMem);
    std::vector<real_t> out(shape.getVolume());
    for (size_t z = 0; z < D; ++z)
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x)
                out[(z * H + y) * W + x] = h.getData()[z * H * P + y * P + x];
    return out;
}
} // namespace

// Isolate the prefilter + getMin: the two things that diverge in the real run.
TEST(FFTParity, PrefilterAndGetMinCPUvsCUDA) {
    CuboidShape shape{80, 60, 20};
    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<real_t> sig(shape.getVolume());
    for (auto& s : sig) s = dist(gen);

    auto& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();

    auto run = [&](const std::string& name) {
        IBackend* b = getBE(name);
        const auto& mem = b->getMemoryManager();
        auto& compute = b->mutableComputeManager();
        size_t W = shape.width, H = shape.height, D = shape.depth;
        size_t P = 2 * (W / 2 + 1);
        std::vector<real_t> padded(D * H * P, 0.0f);
        for (size_t z = 0; z < D; ++z)
            for (size_t y = 0; y < H; ++y)
                for (size_t x = 0; x < W; ++x)
                    padded[z * H * P + y * P + x] = sig[(z * H + y) * W + x];

        RealData g = mem.allocateMemoryOnDeviceRealFFTInPlace(shape);
        RealData f = mem.allocateMemoryOnDeviceRealFFTInPlace(shape);
        mem.memCopy(padded.data(), g.getData(), padded.size() * sizeof(real_t), shape);
        b->sync();

        FFTPlanDescription fReal{PlanDirection::FORWARD, PlanType::REAL, shape, true};
        FFTPlanDescription bReal{PlanDirection::BACKWARD, PlanType::REAL, shape, true};
        compute.initializePlan(fReal);
        compute.initializePlan(bReal);
        ComplexData sieveOTF = mem.allocateMemoryOnDeviceComplex(shape);

        // sigma 0.3 (like prefilter)
        real_t sx = 0.3f, sy = 0.3f, sz = 0.3f;
        // applySieve equivalent: r2c, *GaussianOTF, c2r
        ComplexView fv = f.reinterpret();
        mem.memCopy(g, f);   // f = g (padded zero-copy)
        b->sync();
        compute.fillGaussianOTF(sieveOTF, sx, sy, sz);
        compute.forwardFFT(f, fv);
        compute.complexMultiplication(fv, sieveOTF, fv);
        compute.backwardFFT(fv, f);
        b->sync();

        real_t gmin = compute.getMin(g);
        compute.realClampMin(f, 0.0f, f);
        real_t fmin = compute.getMin(f);
        b->sync();

        // read back f (logical)
        RealData h = mem.moveDataFromDevice(f, hostMem);
        std::vector<real_t> fout(shape.getVolume());
        for (size_t z = 0; z < D; ++z)
            for (size_t y = 0; y < H; ++y)
                for (size_t x = 0; x < W; ++x)
                    fout[(z * H + y) * W + x] = h.getData()[z * H * P + y * P + x];
        return std::make_tuple(gmin, fmin, std::move(fout));
    };

    auto [gminC, fminC, fC] = run("cpu");
    auto [gminU, fminU, fU] = run("cuda");
    double maxField = 0;
    for (size_t i = 0; i < fC.size(); ++i) maxField = std::max(maxField, std::fabs((double)fC[i] - (double)fU[i]));
    std::cout << "[prefilter+getmin] CPU: getMin(g)=" << gminC << " getMin(f)=" << fminC
              << "   CUDA: getMin(g)=" << gminU << " getMin(f)=" << fminU
              << "   field maxDiff=" << maxField << std::endl;
    // dump fields
    {
        FILE* a = std::fopen("/tmp/dolphin/pref_c.bin", "wb"); std::fwrite(fC.data(), sizeof(real_t), fC.size(), a); std::fclose(a);
        FILE* b = std::fopen("/tmp/dolphin/pref_u.bin", "wb"); std::fwrite(fU.data(), sizeof(real_t), fU.size(), b); std::fclose(b);
    }
}

TEST(FFTParity, RoundTripCPUvsCUDA) {
    CuboidShape shape{80, 60, 20};
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<real_t> sig(shape.getVolume());
    for (auto& s : sig) s = dist(gen);

    auto cpu = roundTrip("cpu", shape, sig);
    auto cuda = roundTrip("cuda", shape, sig);

    double maxRTcpu = 0, maxRTcuda = 0, maxDiff = 0;
    for (size_t i = 0; i < sig.size(); ++i) {
        maxRTcpu = std::max(maxRTcpu, std::fabs((double)cpu[i] - (double)sig[i]));
        maxRTcuda = std::max(maxRTcuda, std::fabs((double)cuda[i] - (double)sig[i]));
        maxDiff = std::max(maxDiff, std::fabs((double)cpu[i] - (double)cuda[i]));
    }
    std::cout << "[roundtrip] CPU max|fwd*bwd - id| = " << maxRTcpu
              << "  CUDA max|fwd*bwd - id| = " << maxRTcuda
              << "  CPU-vs-CUDA maxDiff = " << maxDiff << std::endl;

    auto cpuF = fwdSpectrum("cpu", shape, sig);
    auto cudaF = fwdSpectrum("cuda", shape, sig);
    double maxF = 0, relF = 0;
    size_t n = std::min(cpuF.size(), cudaF.size());
    double sumC = 0, sumD = 0;
    for (size_t i = 0; i < n; ++i) {
        maxF = std::max(maxF, std::fabs((double)cpuF[i] - (double)cudaF[i]));
        sumC += (double)cpuF[i] * cpuF[i];
        sumD += (double)cudaF[i] * cudaF[i];
    }
    relF = maxF / (std::sqrt(sumC / n) + 1e-12);
    std::cout << "[fwd spectrum] n=" << n << " CPU-vs-CUDA maxAbsDiff=" << maxF
              << " relToRMS=" << relF
              << " bin0 cpu=" << cpuF[0] << " cuda=" << cudaF[0] << std::endl;
    double sumSig = 0; for (auto s : sig) sumSig += s;
    std::cout << "[fwd spectrum] expected DC(sum)=" << sumSig
              << " cpu/expected=" << (cpuF[0] / sumSig)
              << " cuda/expected=" << (cudaF[0] / sumSig) << std::endl;
}
