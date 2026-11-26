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
#include "dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/MemoryComponent.h"
#include "dolphinbackend/MemoryFootprint.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphin/Logging.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
// Convenience accessor for the protected static helpers.
class TestableStrategy : public StandardDeconvolutionStrategy {
public:
    using StandardDeconvolutionStrategy::resolveMemoryLimit;
};

MemoryBudget makeBudget(size_t host, size_t device, bool deviceIsHost) {
    MemoryBudget budget;
    budget.hostTotal_byte = host;
    budget.deviceTotal_byte = device;
    budget.deviceIsHost = deviceIsHost;
    return budget;
}

// Trivial component reporting fixed host/device bytes for any shape.
class FixedComponent : public MemoryComponent {
public:
    FixedComponent(size_t host, size_t device) : req{host, device} {}
    MemoryRequirement footprint(const CuboidShape&, size_t) const override { return req; }
    std::string name() const override { return "fixed"; }
private:
    MemoryRequirement req;
};
} // namespace

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

// --- resolveMemoryLimit: auto means ~90% of available ------------------------

TEST_F(MemoryTest, AutoLimitUsesSafetyFactor) {
    const size_t available = 10ull * 1024 * 1024 * 1024;
    const size_t expected = static_cast<size_t>(static_cast<double>(available) * AUTO_MEMORY_SAFETY_FACTOR);

    // -1 (default) and 0 are both "auto".
    EXPECT_EQ(TestableStrategy::resolveMemoryLimit(-1.0f, available), expected);
    EXPECT_EQ(TestableStrategy::resolveMemoryLimit(0.0f, available), expected);
    EXPECT_LT(TestableStrategy::resolveMemoryLimit(-1.0f, available), available);
}

TEST_F(MemoryTest, ExplicitLimitIsExactUpperBound) {
    const size_t available = 10ull * 1024 * 1024 * 1024;
    const float gb = 4.0f;
    EXPECT_EQ(TestableStrategy::resolveMemoryLimit(gb, available), static_cast<size_t>(gb * 1024 * 1024 * 1024));
}

TEST_F(MemoryTest, ExplicitLimitNeverExceedsAvailable) {
    const size_t available = 4ull * 1024 * 1024 * 1024;
    // Requesting more than exists must clamp DOWN, never raise the limit.
    EXPECT_EQ(TestableStrategy::resolveMemoryLimit(64.0f, available), available);
}

// --- MemoryBudget::fits: pool accounting --------------------------------------

TEST_F(MemoryTest, FitsChecksBothPoolsIndependently) {
    MemoryBudget budget = makeBudget(100, 100, /*deviceIsHost*/ false);

    EXPECT_TRUE(budget.fits(MemoryRequirement{100, 0}));
    EXPECT_TRUE(budget.fits(MemoryRequirement{0, 100}));
    EXPECT_TRUE(budget.fits(MemoryRequirement{50, 50}));
    EXPECT_FALSE(budget.fits(MemoryRequirement{101, 0}));
    EXPECT_FALSE(budget.fits(MemoryRequirement{0, 101}));
}

TEST_F(MemoryTest, CpuBackendMergesBothPoolsIntoHost) {
    // CPU: device == host, so host + device demand must fit the host pool.
    MemoryBudget budget = makeBudget(100, 100, /*deviceIsHost*/ true);

    EXPECT_TRUE(budget.fits(MemoryRequirement{60, 40}));
    EXPECT_FALSE(budget.fits(MemoryRequirement{60, 50})); // sum 110 > host pool
    // The device pool value is irrelevant on the CPU path.
    MemoryBudget hugeDevice = makeBudget(100, 1000000, true);
    EXPECT_FALSE(hugeDevice.fits(MemoryRequirement{60, 50}));
}

// --- MemoryFootprint: sums the components -------------------------------------

TEST_F(MemoryTest, FootprintSumsAllComponents) {
    FixedComponent hostOnly(10, 0);
    FixedComponent deviceOnly(0, 20);
    FixedComponent mixed(1, 2);

    MemoryFootprint footprint;
    footprint.add(hostOnly);
    footprint.add(deviceOnly);
    footprint.add(mixed);

    MemoryRequirement sum = footprint.total(CuboidShape(4, 4, 4), 1);
    EXPECT_EQ(sum.hostBytes, 11u);
    EXPECT_EQ(sum.deviceBytes, 22u);
}

TEST_F(MemoryTest, FootprintWithBudgetRejectsOversizedShapes) {
    MemoryBudget budget = makeBudget(1000, 1000, false);

    // A component that scales with the shape: small fits, large does not.
    class ScalingComponent : public MemoryComponent {
    public:
        MemoryRequirement footprint(const CuboidShape& s, size_t) const override {
            return MemoryRequirement{0, s.getVolume() * sizeof(real_t)};
        }
        std::string name() const override { return "scaling"; }
    };
    ScalingComponent scaling;
    MemoryFootprint scalingFootprint;
    scalingFootprint.add(scaling);

    EXPECT_TRUE(budget.fits(scalingFootprint.total(CuboidShape(5, 5, 5), 1)));     // 125*4 = 500 bytes
    EXPECT_FALSE(budget.fits(scalingFootprint.total(CuboidShape(32, 32, 32), 1))); // 32768*4 bytes
}

TEST_F(MemoryTest, FootprintBreakdownNamesComponentsAndSums) {
    class NamedComponent : public MemoryComponent {
    public:
        NamedComponent(std::string n, MemoryRequirement r) : n(std::move(n)), r(r) {}
        MemoryRequirement footprint(const CuboidShape&, size_t) const override { return r; }
        std::string name() const override { return n; }
    private:
        std::string n;
        MemoryRequirement r;
    };

    NamedComponent alpha("alpha", MemoryRequirement{100, 0});
    NamedComponent beta("beta", MemoryRequirement{0, 200});
    MemoryFootprint footprint;
    footprint.add(alpha, 2); // two concurrent instances
    footprint.add(beta);

    const std::string breakdown = footprint.breakdown(CuboidShape(1, 1, 1), 1);
    EXPECT_NE(breakdown.find("alpha"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("(x2)"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("beta"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("total"), std::string::npos) << breakdown;

    const MemoryRequirement sum = footprint.total(CuboidShape(1, 1, 1), 1);
    EXPECT_EQ(sum.hostBytes, 200u);
    EXPECT_EQ(sum.deviceBytes, 200u);
}

TEST_F(MemoryTest, SetMemoryLimitDefaultIsNoOp) {
    // A component without its own tracking accepts any cap without effect.
    class NoTrackingComponent : public MemoryComponent {
    public:
        MemoryRequirement footprint(const CuboidShape&, size_t) const override { return MemoryRequirement{1, 1}; }
        std::string name() const override { return "no tracking"; }
    };
    NoTrackingComponent component;
    EXPECT_NO_THROW(component.setMemoryLimit(MemoryRequirement{10, 10}));
}

// --- Task-aware components: min(threads, numTasks) instances -------------------
// A system holding one resource set per thread can never have more sets in
// flight than there are tasks (cubes) in the plan.

TEST_F(MemoryTest, IOStagingCapsInstancesAtTaskCount) {
    const CuboidShape shape(4, 4, 4);
    const size_t volumeBytes = shape.getVolume() * sizeof(real_t);

    IOStagingMemoryComponent staging(/*ioThreads=*/4);

    // One task: exactly one staging set, regardless of the io thread count.
    const MemoryRequirement one = staging.footprint(shape, 1);
    EXPECT_EQ(one.hostBytes, volumeBytes);
    EXPECT_EQ(one.deviceBytes, 2 * volumeBytes);

    // Fewer tasks than io threads: one set per task.
    const MemoryRequirement two = staging.footprint(shape, 2);
    EXPECT_EQ(two.hostBytes, 2 * volumeBytes);
    EXPECT_EQ(two.deviceBytes, 4 * volumeBytes);

    // More tasks than io threads: capped at the thread count.
    const MemoryRequirement many = staging.footprint(shape, 8);
    EXPECT_EQ(many.hostBytes, 4 * volumeBytes);
    EXPECT_EQ(many.deviceBytes, 8 * volumeBytes);
}

// A backend manager whose only modeled part is the FFT plan answer: the
// algorithm's memory component must funnel the plan set through it.
class FakeBackendManager : public IBackendManager {
public:
    explicit FakeBackendManager(size_t fftBytes) : fftBytes(fftBytes) {}

    void init(LogCallback) override {}
    IBackend& createBackendForCurrentThread(const BackendConfig&) override { throw std::runtime_error("not used"); }
    IBackend& createBackendSharedMemoryForCurrentThread(IBackend&, const BackendConfig&) override { throw std::runtime_error("not used"); }
    size_t getNumberDevices() const override { return 1; }
    void setThreadDistribution(ThreadConfig&, BackendConfig&, BackendConfig&) override {}
    MemoryComponent& getMemoryComponent() override { return component; }

    // Answers for the whole set; records what the algorithm asked for.
    size_t getFFTPlanMemoryBytes(const std::vector<FFTPlanDescription>& descriptions) const override {
        queriedDescriptions = descriptions;
        return fftBytes;
    }

    // Nothing to report at the manager level for these tests.
    class EmptyComponent : public MemoryComponent {
    public:
        MemoryRequirement footprint(const CuboidShape&, size_t) const override { return {0, 0}; }
        std::string name() const override { return "fake backend"; }
    } component;

    mutable std::vector<FFTPlanDescription> queriedDescriptions;
    size_t fftBytes;
};

TEST_F(MemoryTest, AlgorithmComponentFunnelsPlanMemoryThroughBackendManager) {
    const CuboidShape shape(4, 4, 4);
    auto config = DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm("RichardsonLucy");
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = DeconvolutionAlgorithmFactory::getInstance().createShared(*config);
    ASSERT_NE(algorithm, nullptr);

    const size_t fftBytes = 777;
    FakeBackendManager backend(fftBytes);
    const size_t workingSetBytes = algorithm->getWorkingSetBytes(shape);

    // Host plane, three worker threads.
    auto onHost = algorithm->getMemoryComponent(backend, /*onDevice=*/false, /*workerThreads=*/3);

    // One task: one scratch set plus the backend's answer for the plan set.
    const MemoryRequirement one = onHost->footprint(shape, 1);
    EXPECT_EQ(one.hostBytes, workingSetBytes + fftBytes);
    EXPECT_EQ(one.deviceBytes, 0u);

    // Both terms carry the same concurrency: capped at the worker thread count.
    const MemoryRequirement many = onHost->footprint(shape, 10);
    EXPECT_EQ(many.hostBytes, 3 * (workingSetBytes + fftBytes));
    EXPECT_EQ(many.deviceBytes, 0u);

    // The algorithm declared its plan set (RL: out-of-place forward r2c,
    // in-place backward c2r, and the second-pass r2c in-place on the c
    // buffer), which is exactly what the backend was asked for.
    const std::vector<FFTPlanDescription> declared = algorithm->getFFTPlanDescriptions(shape);
    EXPECT_EQ(declared.size(), 3u);
    ASSERT_EQ(backend.queriedDescriptions.size(), declared.size());
    for (size_t i = 0; i < declared.size(); ++i)
        EXPECT_TRUE(backend.queriedDescriptions[i] == declared[i]) << "plan description mismatch at " << i;
}

TEST_F(MemoryTest, AlgorithmComponentUsesRealCpuBackendPlanAnswer) {
    // The real CPU manager answers the plan query; with no FFT plan memory
    // and no scratch, an empty-plan algorithm must report exactly zero.
    const CuboidShape shape(32, 16, 8);
    class NoPlansAlgorithm : public DeconvolutionAlgorithm {
    public:
        void configure(const DeconvolutionConfig&) override {}
        void init(const CuboidShape&) override { initialized = true; }
        bool isInitialized() const override { return initialized; }
        void deconvolve(const ComplexData&, RealData&, RealData&, const DeconvolutionRunContext&) override {}
        size_t getWorkingSetBytes(const CuboidShape&) const override { return 0; }
    private:
        std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const override { return std::make_unique<NoPlansAlgorithm>(*this); }
        bool initialized = false;
    };

    NoPlansAlgorithm algorithm;
    IBackendManager& cpu = BackendFactory::getBackendManagerStatic(HOST_BACKEND);
    EXPECT_EQ(cpu.getFFTPlanMemoryBytes({}), 0u);

    auto component = algorithm.getMemoryComponent(cpu, /*onDevice=*/false, /*workerThreads=*/2);
    const MemoryRequirement req = component->footprint(shape, 2);
    EXPECT_EQ(req.hostBytes, 0u);
    EXPECT_EQ(req.deviceBytes, 0u);
    EXPECT_NE(component->name().find("fft plans"), std::string::npos);
}
