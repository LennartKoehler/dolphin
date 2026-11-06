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
#include "dolphinbackend/MemoryComponent.h"
#include "dolphinbackend/MemoryFootprint.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphin/Logging.h"
#include <algorithm>
#include <string>
#include <utility>

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
    MemoryRequirement footprint(const CuboidShape&) const override { return req; }
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

    MemoryRequirement sum = footprint.total(CuboidShape(4, 4, 4));
    EXPECT_EQ(sum.hostBytes, 11u);
    EXPECT_EQ(sum.deviceBytes, 22u);
}

TEST_F(MemoryTest, FootprintWithBudgetRejectsOversizedShapes) {
    MemoryBudget budget = makeBudget(1000, 1000, false);

    // A component that scales with the shape: small fits, large does not.
    class ScalingComponent : public MemoryComponent {
    public:
        MemoryRequirement footprint(const CuboidShape& s) const override {
            return MemoryRequirement{0, s.getVolume() * sizeof(real_t)};
        }
        std::string name() const override { return "scaling"; }
    };
    ScalingComponent scaling;
    MemoryFootprint scalingFootprint;
    scalingFootprint.add(scaling);

    EXPECT_TRUE(budget.fits(scalingFootprint.total(CuboidShape(5, 5, 5))));     // 125*4 = 500 bytes
    EXPECT_FALSE(budget.fits(scalingFootprint.total(CuboidShape(32, 32, 32)))); // 32768*4 bytes
}

TEST_F(MemoryTest, FootprintBreakdownNamesComponentsAndSums) {
    class NamedComponent : public MemoryComponent {
    public:
        NamedComponent(std::string n, MemoryRequirement r) : n(std::move(n)), r(r) {}
        MemoryRequirement footprint(const CuboidShape&) const override { return r; }
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

    const std::string breakdown = footprint.breakdown(CuboidShape(1, 1, 1));
    EXPECT_NE(breakdown.find("alpha"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("(x2)"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("beta"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("total"), std::string::npos) << breakdown;

    const MemoryRequirement sum = footprint.total(CuboidShape(1, 1, 1));
    EXPECT_EQ(sum.hostBytes, 200u);
    EXPECT_EQ(sum.deviceBytes, 200u);
}

TEST_F(MemoryTest, SetMemoryLimitDefaultIsNoOp) {
    // A component without its own tracking accepts any cap without effect.
    class NoTrackingComponent : public MemoryComponent {
    public:
        MemoryRequirement footprint(const CuboidShape&) const override { return MemoryRequirement{1, 1}; }
        std::string name() const override { return "no tracking"; }
    };
    NoTrackingComponent component;
    EXPECT_NO_THROW(component.setMemoryLimit(MemoryRequirement{10, 10}));
}
