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
#include "dolphin/deconvolution/algorithms/PoissonMLECore.h"
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
    IBackend& createBackend(const BackendConfig&) override { throw std::runtime_error("not used"); }
    size_t getNumberDevices() const override { return 1; }
    void setThreadDistribution(ThreadConfig&, BackendConfig&, BackendConfig&) override {}

    // Answers for the whole set; records what the algorithm asked for. The
    // thread count is part of the query (the backend folds its per-instance
    // semantics in); this fake ignores it and answers a fixed value.
    size_t getFFTPlanMemoryBytes(const std::vector<FFTPlanDescription>& descriptions, const size_t workerThreads) const override {
        (void)workerThreads;
        queriedDescriptions = descriptions;
        return fftBytes;
    }

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
    // RL's own component knowledge: one complex working buffer (2 real_t per
    // element) per instance.
    const size_t workingSetBytes = 2 * shape.getVolume() * sizeof(real_t);

    // Host plane, three worker threads.
    auto onHost = algorithm->getMemoryComponent(backend, /*workerThreads=*/3);

    // One task: one scratch set plus the backend's answer for the plan set.
    // Components always report the device plane; MemoryBudget::fits folds it
    // into the host pool on a CPU backend (device == host).
    const MemoryRequirement one = onHost->footprint(shape, 1);
    EXPECT_EQ(one.deviceBytes, workingSetBytes + fftBytes);
    EXPECT_EQ(one.hostBytes, 0u);

    // The scratch carries the per-thread concurrency (3 instances); the plan
    // answer does NOT: the backend folds its own thread semantics in (this
    // fake answers a flat value; CUDA multiplies by its worker count inside).
    const MemoryRequirement many = onHost->footprint(shape, 10);
    EXPECT_EQ(many.deviceBytes, 3 * workingSetBytes + fftBytes);
    EXPECT_EQ(many.hostBytes, 0u);

    // The component asked the backend for RL's plan set (out-of-place forward
    // r2c, in-place backward c2r, and the second-pass r2c in-place on the c
    // buffer) — and for nothing else.
    const std::vector<FFTPlanDescription> expectedPlans = {
        {PlanDirection::FORWARD, PlanType::REAL, shape, false},
        {PlanDirection::BACKWARD, PlanType::REAL, shape, true},
        {PlanDirection::FORWARD, PlanType::REAL, shape, true},
    };
    ASSERT_EQ(backend.queriedDescriptions.size(), expectedPlans.size());
    for (size_t i = 0; i < expectedPlans.size(); ++i)
        EXPECT_TRUE(backend.queriedDescriptions[i] == expectedPlans[i]) << "plan description mismatch at " << i;
}

TEST_F(MemoryTest, AlgorithmComponentUsesRealCpuBackendPlanAnswer) {
    // The real CPU manager answers the plan query with 0: FFTW planning holds
    // no persistent memory (the transient is freed after planning). The
    // composed component of the classic PoissonMLE is therefore exactly its
    // scratch: one real in-place-FFT buffer per instance.
    const CuboidShape shape(32, 16, 8);
    auto config = DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm("PoissonMLE");
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = DeconvolutionAlgorithmFactory::getInstance().createShared(*config);
    ASSERT_NE(algorithm, nullptr);

    IBackendManager& cpu = BackendFactory::getBackendManagerStatic(HOST_BACKEND);
    EXPECT_EQ(cpu.getFFTPlanMemoryBytes(PoissonMLECore::planDescriptions(shape), 4), 0u);

    auto component = algorithm->getMemoryComponent(cpu, /*workerThreads=*/2);
    const MemoryRequirement one = component->footprint(shape, 1);
    EXPECT_EQ(one.deviceBytes, shape.getVolume() * sizeof(real_t));
    EXPECT_EQ(one.hostBytes, 0u);
    EXPECT_NE(component->name().find("fft plans"), std::string::npos);

    // The scratch carries the per-thread concurrency: two instances with two
    // tasks, capped at the task count with fewer tasks.
    EXPECT_EQ(component->footprint(shape, 2).deviceBytes, 2 * shape.getVolume() * sizeof(real_t));
    EXPECT_EQ(component->footprint(shape, 1).deviceBytes, shape.getVolume() * sizeof(real_t));
}
