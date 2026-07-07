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
#include "dolphinbackend/CuboidShape.h"
#include "dolphin/Logging.h"
#include <stdexcept>

namespace {
// Convenience accessor for the protected static helper.
class TestableStrategy : public StandardDeconvolutionStrategy {
public:
    using StandardDeconvolutionStrategy::computeMaxMemoryPerCube;
};

// Returns the bytes available per cube when the FFT workspace is zero.
// With fftWorkspace == 0 the convergence loop in computeMaxMemoryPerCube is
// idempotent, so the result is just availableMemory / threadallocations.
size_t expectedPerCubeNoFFT(size_t availableMemory,
                            size_t ioThreads,
                            size_t workerThreads,
                            size_t multiplier) {
    size_t threadallocations = ioThreads * 3 + workerThreads * multiplier;
    return availableMemory / threadallocations;
}
} // namespace

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

// --- No FFT workspace overhead: result is a pure division -------------------

TEST_F(MemoryTest, NoFFTWorkspaceSimpleDivision) {
    const size_t available = 10ull * 1024 * 1024 * 1024; // 10 GB
    const size_t ioThreads = 2;
    const size_t workerThreads = 4;
    const size_t multiplier = 2; // e.g. Richardson-Lucy

    auto estimator = [](const CuboidShape&) { return size_t(0); };

    size_t result = TestableStrategy::computeMaxMemoryPerCube(
        available, ioThreads, workerThreads, multiplier, estimator);

    EXPECT_NEAR(result, expectedPerCubeNoFFT(available, ioThreads, workerThreads, multiplier), 100);
}

TEST_F(MemoryTest, SingleWorkerSingleIO) {
    const size_t available = 8ull * 1024 * 1024 * 1024;
    auto estimator = [](const CuboidShape&) { return size_t(0); };

    size_t result = TestableStrategy::computeMaxMemoryPerCube(
        available, /*ioThreads*/ 1, /*workerThreads*/ 1, /*multiplier*/ 1, estimator);

    // threadallocations = 1*3 + 1*1 = 4
    EXPECT_EQ(result, available / 4);
}

// --- Result is always strictly below the available memory ------------------

TEST_F(MemoryTest, ResultDoesNotExceedAvailableMemory) {
    const size_t available = 4ull * 1024 * 1024 * 1024;
    auto estimator = [](const CuboidShape& shape) {
        return shape.getVolume() * sizeof(complex_t);
    };

    size_t result = TestableStrategy::computeMaxMemoryPerCube(
        available, 2, 4, 2, estimator);

    EXPECT_LT(result, available);
}

TEST_F(MemoryTest, ResultIsPositiveForReasonableMemory) {
    const size_t available = 16ull * 1024 * 1024 * 1024;
    auto estimator = [](const CuboidShape&) { return size_t(0); };

    size_t result = TestableStrategy::computeMaxMemoryPerCube(
        available, 1, 1, 1, estimator);

    EXPECT_GT(result, 0u);
}

TEST_F(MemoryTest, ResultDoesNotExceedAvailableMemoryStrict) {
    const size_t available = 4ull * 1024 * 1024 * 1024;
    auto estimator = [](const CuboidShape& shape) {
        return shape.getVolume() * sizeof(complex_t);
    };

    size_t workerThreads = 2;
    size_t ioThreads = 3;
    size_t algorithmMemoryMultiplier = 2;

    size_t result = TestableStrategy::computeMaxMemoryPerCube(
        available, ioThreads, workerThreads, algorithmMemoryMultiplier, estimator);

    size_t workerAllocations = workerThreads * algorithmMemoryMultiplier;

    int ioCopies = 3; //image, psf, result, but psf only allocated once in total
    size_t ioAllocations = ioThreads * ioCopies;

    size_t threadallocations = ioAllocations + workerAllocations;

    EXPECT_LT(result * threadallocations, available);
}

// --- More threads => smaller per-cube budget --------------------------------

TEST_F(MemoryTest, MoreThreadsYieldSmallerCube) {
    const size_t available = 10ull * 1024 * 1024 * 1024;
    auto estimator = [](const CuboidShape&) { return size_t(0); };

    size_t few = TestableStrategy::computeMaxMemoryPerCube(
        available, 1, 1, 1, estimator);
    size_t many = TestableStrategy::computeMaxMemoryPerCube(
        available, 4, 8, 2, estimator);

    EXPECT_GT(few, many);
}

// --- Higher algorithm memory multiplier => smaller per-cube budget ----------

TEST_F(MemoryTest, HigherMultiplierYieldsSmallerCube) {
    const size_t available = 10ull * 1024 * 1024 * 1024;
    auto estimator = [](const CuboidShape&) { return size_t(0); };

    size_t lowMult = TestableStrategy::computeMaxMemoryPerCube(
        available, 2, 4, 1, estimator);
    size_t highMult = TestableStrategy::computeMaxMemoryPerCube(
        available, 2, 4, 4, estimator);

    EXPECT_GT(lowMult, highMult);
}

// --- FFT workspace reduces the per-cube budget ------------------------------

TEST_F(MemoryTest, FFTWorkspaceReducesBudget) {
    const size_t available = 10ull * 1024 * 1024 * 1024;
    const size_t ioThreads = 2;
    const size_t workerThreads = 4;
    const size_t multiplier = 2;

    auto noFFT = [](const CuboidShape&) { return size_t(0); };
    auto withFFT = [workerThreads, available](const CuboidShape&) {
        // A modest but non-zero workspace, well below the available memory.
        return available / (workerThreads * 10);
    };

    size_t withoutFFT = TestableStrategy::computeMaxMemoryPerCube(
        available, ioThreads, workerThreads, multiplier, noFFT);
    size_t withFFTs = TestableStrategy::computeMaxMemoryPerCube(
        available, ioThreads, workerThreads, multiplier, withFFT);

    EXPECT_LT(withFFTs, withoutFFT);
}

// --- Zero available memory -------------------------------------------------

TEST_F(MemoryTest, ZeroAvailableMemoryReturnsZero) {
    auto estimator = [](const CuboidShape&) { return size_t(0); };

    size_t result = TestableStrategy::computeMaxMemoryPerCube(
        /*available*/ 0, 1, 1, 1, estimator);

    // cubeVolume becomes 0 -> loop breaks immediately -> 0 / threadallocations
    EXPECT_EQ(result, 0u);
}
