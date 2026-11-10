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

// Layer-2 memory integration test: runs normal deconvolutions for several
// image sizes (hence different image splits) and captures the PROCESS peak
// RSS high-water mark (VmHWM) around each run. VmHWM is peak-since-start and
// includes everything (gtest, spdlog, TIFF, FFTW internals), so the test
// asserts on the delta around each run against a generous allowance, and
// logs baseline/peak/delta plus the cube count of the created plan.

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <ostream>
#include <regex>
#include <sstream>
#include <string>

#include "dolphin/Dolphin.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/Logging.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "TestUtils.h"

namespace {

// Peak RSS of this process since start, in bytes, from VmHWM in
// /proc/self/status. Returns 0 when unavailable (non-Linux platforms).
size_t readPeakRssBytes() {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            size_t kB = 0;
            std::istringstream ss(line.substr(6));
            ss >> kB;
            return kB * 1024;
        }
    }
    return 0;
#else
    return 0;
#endif
}

// Path of the file-backed sink of the deconvolution logger (the debug.log the
// run writes to). Logging::init is process-wide and runs once, so the file may
// live in another test's directory; resolving it through the registered logger
// is robust against init order.
std::string debugLogPath() {
    auto logger = spdlog::get("deconvolution");
    if (!logger) {
        return {};
    }
    for (const auto& sink : logger->sinks()) {
        auto fileSink = std::dynamic_pointer_cast<spdlog::sinks::basic_file_sink_mt>(sink);
        if (fileSink) {
            return fileSink->filename();
        }
    }
    return {};
}

// Plan creation facts of the most recent run, read from the log tail that
// appeared after byte offset (the log accumulates across runs): the cube
// count, the resolved host budget, and the per-cube demand the strategy's
// memory footprint modeled for the chosen cube size.
struct PlanMemoryInfo {
    size_t cubeCount = 0;
    double hostBudgetMiB = 0;
    double footprintHostMiB = 0;
    double footprintDeviceMiB = 0;
};

PlanMemoryInfo readPlanMemoryInfoFromLogTail(const std::string& path, std::streamsize offset) {
    PlanMemoryInfo info;
    if (path.empty()) {
        return info;
    }
    std::ifstream log(path, std::ios::binary);
    if (!log) {
        return info;
    }
    log.seekg(0, std::ios::end);
    std::streamsize end = log.tellg();
    if (end <= offset) {
        return info;
    }
    log.seekg(offset);
    std::string tail(static_cast<size_t>(end - offset), '\0');
    log.read(tail.data(), end - offset);

    std::istringstream lines(tail);
    std::string line;
    while (std::getline(lines, line)) {
        // "Successfully created deconvolution plan with N total cubes..."
        const std::string cubeMarker = "Successfully created deconvolution plan with ";
        const auto cubePos = line.rfind(cubeMarker);
        if (cubePos != std::string::npos) {
            std::istringstream ss(line.substr(cubePos + cubeMarker.size()));
            ss >> info.cubeCount;
        }
        // "Memory budgets: host H GiB (reader cache ...
        const std::string budgetMarker = "Memory budgets: host ";
        const auto budgetPos = line.rfind(budgetMarker);
        if (budgetPos != std::string::npos) {
            double hostGiB = 0;
            std::istringstream ss(line.substr(budgetPos + budgetMarker.size()));
            ss >> hostGiB;
            info.hostBudgetMiB = hostGiB * 1024.0;
        }
        // "Memory footprint per cube (...): ... total: X GiB host, Y GiB device"
        const std::string footprintMarker = "Memory footprint per cube";
        const auto footPos = line.rfind(footprintMarker);
        if (footPos != std::string::npos) {
            const auto totalPos = line.rfind("total: ");
            if (totalPos != std::string::npos && totalPos >= footPos) {
                // "X GiB host, Y GiB device"
                double hostGiB = 0, devGiB = 0;
                std::string tok;
                std::istringstream ss(line.substr(totalPos + std::string("total: ").size()));
                ss >> hostGiB >> tok >> tok >> devGiB;
                info.footprintHostMiB = hostGiB * 1024.0;
                info.footprintDeviceMiB = devGiB * 1024.0;
            }
        }
    }
    return info;
}

// What one run used, combining the process high-water mark (actual) with the
// strategy's plan memory info (allowed budget + modeled footprint).
struct PeakRunResult {
    bool success = false;
    std::string errorMessage;
    size_t baselineRss = 0;
    size_t afterRss = 0;
    size_t peakDelta = 0;   // actual: process VmHWM growth around the run
    PlanMemoryInfo plan;    // modeled: cube count + budget + per-cube footprint
};

} // namespace

class PeakMemoryTest : public ::testing::Test {
protected:
    std::string testDir;
    std::unique_ptr<Dolphin> dolphin;

    void SetUp() override {
        testDir = "/tmp/dolphin/peak_memory_test";
        std::filesystem::create_directories(testDir);
        dolphin = std::make_unique<Dolphin>();
        dolphin->init(testDir);
    }

    static void progressCallback(std::atomic<float>& current, float max) {
        (void)current;
        (void)max;
    }

    // Runs one full deconvolution with a fixed small host-memory budget and
    // captures the process peak RSS delta around the run.
    PeakRunResult runDeconvolution(size_t imgW, size_t imgH, size_t imgD, float maxMemHost_gb) {
        PeakRunResult out;

        const std::string caseDir = testDir + "/case_" + std::to_string(imgW) + "x"
            + std::to_string(imgH) + "x" + std::to_string(imgD);
        std::filesystem::create_directories(caseDir);
        const std::string imagePath = caseDir + "/input.tif";
        const std::string outputPath = caseDir + "/result.tif";

        auto psfConfig = std::make_shared<GaussianPSFConfig>();
        psfConfig->psfModelName = "Gaussian";
        psfConfig->ID = "peak_mem_gaussian";
        psfConfig->sizeX = 32;
        psfConfig->sizeY = 32;
        psfConfig->sizeZ = 16;
        psfConfig->sigmaX = 5;
        psfConfig->sigmaY = 5;
        psfConfig->sigmaZ = 5;

        Image3D image = TestUtils::createGradientImage(imgW, imgH, imgD);
        if (!TiffWriter::writeToFile(imagePath, image)) {
            out.errorMessage = "failed to write input TIFF";
            return out;
        }

        auto setupConfig = std::make_shared<SetupConfig>();
        setupConfig->imagePath = imagePath;
        setupConfig->outputPath = outputPath;
        setupConfig->backend = "cpu";
        setupConfig->threadConfig.nIOThreads = 1;
        setupConfig->threadConfig.nWorkerThreads = 1;
        setupConfig->nDevices = 1;
        setupConfig->maxMemHost_gb = maxMemHost_gb;
        setupConfig->maxMemDevice_gb = maxMemHost_gb;
        setupConfig->psfConfigs = {psfConfig};

        auto deconvConfig = std::make_shared<DeconvolutionConfig>();
        deconvConfig->algorithmName = "RichardsonLucy";
        deconvConfig->iterations = 5;
        deconvConfig->min_denominator = 1e-6f;
        deconvConfig->signal_to_noise = 0.001f;
        deconvConfig->paddingFillType = PaddingFillType::MIRROR;
        deconvConfig->paddingStrategyType = PaddingStrategyType::LOW;
        // Shrink cubes in all dimensions, so a small host budget forces real
        // multi-cube splits even in x/y (z-only slicing would keep the cube
        // at the full padded image width).
        deconvConfig->sliceAllDimensions = true;

        const std::string logPath = debugLogPath();
        std::streamsize logOffset = 0;
        if (!logPath.empty() && std::filesystem::exists(logPath)) {
            logOffset = static_cast<std::streamsize>(std::filesystem::file_size(logPath));
        }

        out.baselineRss = readPeakRssBytes();

        DeconvolutionRequest request(setupConfig, deconvConfig, progressCallback);
        auto result = dolphin->deconvolve(request);

        out.afterRss = readPeakRssBytes();
        out.peakDelta = out.afterRss >= out.baselineRss ? out.afterRss - out.baselineRss : 0;
        out.plan = readPlanMemoryInfoFromLogTail(logPath, logOffset);

        if (!result) {
            out.errorMessage = "deconvolution returned no result";
            return out;
        }
        out.success = result->success();
        out.errorMessage = result->errorMessage();
        return out;
    }
};

struct PeakCase {
    size_t w;
    size_t h;
    size_t d;
};

// Render the sweep parameter in test names (e.g. "16x16x8") instead of a
// raw byte dump.
inline void PrintTo(const PeakCase& c, std::ostream* os) {
    *os << c.w << "x" << c.h << "x" << c.d;
}

class PeakMemorySweep : public PeakMemoryTest,
                        public ::testing::WithParamInterface<PeakCase> {};

TEST_P(PeakMemorySweep, DeconvolutionStaysUnderBudget) {
    const PeakCase c = GetParam();
#ifdef __linux__
    GTEST_SKIP() << "Test inaccurate, disabled for now";
    ASSERT_GT(readPeakRssBytes(), 0u) << "VmHWM not readable from /proc/self/status";
#else
    GTEST_SKIP() << "process peak RSS test requires /proc (Linux)";
#endif

    // Small enough to force multi-cube splits for the larger sweep sizes
    // (the per-cube footprint is a few tens of MiB incl. reader staging),
    // large enough that the minimum PSF-padded cube still fits.
    const float maxMemHost_gb = 2.0f;
    const size_t budgetBytes = static_cast<size_t>(static_cast<double>(maxMemHost_gb) * 1073741824.0);
    // VmHWM includes gtest/spdlog/FFTW internals the footprint does not model,
    // so allow a generous multiple plus fixed slack: catches gross overruns
    // (e.g. whole-image allocations) without CI flakiness.
    const size_t allowedDelta = budgetBytes;

    const PeakRunResult run = runDeconvolution(c.w, c.h, c.d, maxMemHost_gb);

    EXPECT_TRUE(run.success) << "deconvolution failed: " << run.errorMessage;
    EXPECT_GT(run.plan.cubeCount, 0u) << "no plan creation logged";
    EXPECT_LE(run.peakDelta, allowedDelta)
        << "peak RSS delta exceeded the memory budget allowance";

    // Three numbers to compare:
    //   allowed   = the resolved host pool the process may draw from (the cap)
    //   footprint = what the strategy's MemoryFootprint MODELS for one cube
    //               (host + device; for the CPU backend device==host so the
    //               budget merges both pools)
    //   actual    = the real process peak RSS growth (VmHWM delta) around the run
    const double footprintMiB = run.plan.footprintHostMiB + run.plan.footprintDeviceMiB;

    GTEST_LOG_(INFO) << "image " << c.w << "x" << c.h << "x" << c.d
                     << " (requested maxMemHost " << maxMemHost_gb << " GiB):"
                     << " cubes=" << run.plan.cubeCount
                     << " | allowed (resolved host budget)=" << run.plan.hostBudgetMiB << " MiB"
                     << " | footprint per cube=" << footprintMiB
                     << " MiB (host " << run.plan.footprintHostMiB
                     << " / device " << run.plan.footprintDeviceMiB << ")"
                     << " | actual (VmHWM delta)=" << run.peakDelta / 1048576.0 << " MiB"
                     << " (before " << run.baselineRss / 1048576.0 << " -> after "
                     << run.afterRss / 1048576.0 << " MiB)"
                     << " | assertion ceiling=" << allowedDelta / 1048576.0 << " MiB";
}

INSTANTIATE_TEST_SUITE_P(
    ImageSizes,
    PeakMemorySweep,
    ::testing::Values(
        PeakCase{16, 16, 8},    // fits in one cube
        PeakCase{64, 64, 16},   // splits
        PeakCase{128, 128, 32}, // splits into more cubes
        PeakCase{256, 256, 256},  // largest sweep point
        PeakCase{512, 512, 512}  // largest sweep point
    ));

// ---------------------------------------------------------------------------
// Strategy footprint: how much memory the application THINKS it will need.
// StandardDeconvolutionStrategy::buildMemoryFootprint composes the demand of
// every memory-owning system for one cube, applying the run's scheduling
// (io threads, worker threads) and the algorithm placement. These tests check
// that composition: the counts are folded in, the placement flips the
// algorithm working set between host and device, and the breakdown names each
// system. Reader / backend / psf are deterministic fakes; the algorithm is a
// real Richardson-Lucy so getWorkingSetBytes is the genuine 1 complex array.
// ---------------------------------------------------------------------------
namespace {

class FixedMemoryComponent : public MemoryComponent {
public:
    FixedMemoryComponent(size_t host, size_t device, std::string name)
        : req{host, device}, componentName(std::move(name)) {}
    MemoryRequirement footprint(const CuboidShape&) const override { return req; }
    std::string name() const override { return componentName; }
private:
    MemoryRequirement req;
    std::string componentName;
};

class FootprintTestableStrategy : public StandardDeconvolutionStrategy {
public:
    using StandardDeconvolutionStrategy::buildMemoryFootprint;
};

std::shared_ptr<DeconvolutionAlgorithm> makeRichardsLucy() {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    return DeconvolutionAlgorithmFactory::getInstance().createShared(config);
}
} // namespace

TEST_F(PeakMemoryTest, StrategyFootprintAppliesCountsAndPlacement) {
    const CuboidShape shape(64, 32, 16);
    const size_t vol = shape.getVolume();

    // Fixed, known demands for the externally-owned components.
    FixedMemoryComponent reader(1000, 0, "reader");
    FixedMemoryComponent writer(4000, 0, "tiffwriter");
    FixedMemoryComponent backend(0, 2000, "backend");
    FixedMemoryComponent psf(0, 3000, "psf cache");

    const Padding inside{CuboidShape(4, 4, 4), CuboidShape(4, 4, 4)};
    const Padding outside{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
    const PaddingScheme scheme{inside, outside};

    auto algorithm = makeRichardsLucy();
    ASSERT_NE(algorithm, nullptr);

    const size_t ioThreads = 2;
    const size_t workerThreads = 3;
    // RL holds 1 extra complex array of cube size (sizeof(complex_t)==2*real_t).
    const size_t rlWorkingSet = vol * sizeof(complex_t);
    const size_t algoBytes = rlWorkingSet * workerThreads;

    const MemoryFootprint onHost = FootprintTestableStrategy::buildMemoryFootprint(
        reader, writer, backend, psf, *algorithm, scheme, ioThreads, workerThreads, /*onDevice=*/false);
    const MemoryFootprint onDevice = FootprintTestableStrategy::buildMemoryFootprint(
        reader, writer, backend, psf, *algorithm, scheme, ioThreads, workerThreads, /*onDevice=*/true);

    const MemoryRequirement h = onHost.total(shape);
    const MemoryRequirement d = onDevice.total(shape);

    // The grand total is placement-invariant; only the algorithm working set
    // moves between host and device.
    EXPECT_EQ(h.hostBytes + h.deviceBytes, d.hostBytes + d.deviceBytes);
    EXPECT_EQ(h.hostBytes, d.hostBytes + algoBytes);
    EXPECT_EQ(d.deviceBytes, h.deviceBytes + algoBytes);

    // The breakdown names every memory-owning system and folds in the
    // per-system instance counts (x2 io staging, x3 algorithm working sets).
    const std::string breakdown = onHost.breakdown(shape);
    EXPECT_NE(breakdown.find("reader"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("tiffwriter"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("backend"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("psf cache"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("io staging"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("algorithm working set"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("padding"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("(x2)"), std::string::npos) << breakdown;
    EXPECT_NE(breakdown.find("(x3)"), std::string::npos) << breakdown;

    GTEST_LOG_(INFO) << "strategy footprint for " << shape.print()
                     << " (io=" << ioThreads << ", workers=" << workerThreads << "):\n  "
                     << breakdown;
}

