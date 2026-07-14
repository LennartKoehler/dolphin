#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <tiffio.h>

#include <spdlog/spdlog.h>

#include "dolphin_image/Image3D.h"
#include "dolphin_image/ImageMetaData.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/Types/BoxCoord.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// ParallelZSliceReader — reads a subimage with z-slice parallelism.
// Each z-slice is an independent IFD; N threads each read a disjoint
// range of z-slices using separate TIFF handles.
// ---------------------------------------------------------------------------
class ParallelZSliceReader {
public:
    ParallelZSliceReader(const std::string& filename, const ImageMetaData& metaData, size_t numThreads)
        : metaData_(metaData), numThreads_(numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            TIFF* tif = TIFFOpen(filename.c_str(), "r");
            if (!tif) {
                for (TIFF* h : handles_) {
                    if (h) TIFFClose(h);
                }
                throw std::runtime_error("Cannot open TIFF: " + filename);
            }
            handles_.push_back(tif);
            available_.push(tif);
        }
    }

    ~ParallelZSliceReader() {
        for (TIFF* tif : handles_) {
            if (tif) TIFFClose(tif);
        }
    }

    ParallelZSliceReader(const ParallelZSliceReader&) = delete;
    ParallelZSliceReader& operator=(const ParallelZSliceReader&) = delete;

    Image3D getSubimage(const BoxCoord& box) {
        Image3D image(box.dimensions, -1.0f);

        TIFF* probe = acquire();
        tsize_t scanlineSize = TIFFScanlineSize(probe);
        release(probe);

        if (scanlineSize <= 0) {
            throw std::runtime_error("Invalid scanline size");
        }

        size_t totalSlices = box.dimensions.depth;
        size_t slicesPerThread = (totalSlices + numThreads_ - 1) / numThreads_;

        std::vector<std::future<void>> futures;
        futures.reserve(numThreads_);

        for (size_t t = 0; t < numThreads_; ++t) {
            size_t zStart = t * slicesPerThread;
            size_t zEnd = std::min(zStart + slicesPerThread, totalSlices);
            if (zStart >= zEnd) break;

            futures.push_back(std::async(std::launch::async,
                [this, &image, &box, zStart, zEnd, scanlineSize]() {
                    this->readSliceRange(image, box, zStart, zEnd, scanlineSize);
                }));
        }

        for (auto& f : futures) {
            f.get();
        }

        return image;
    }

private:
    TIFF* acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !available_.empty(); });
        TIFF* tif = available_.front();
        available_.pop();
        return tif;
    }

    void release(TIFF* tif) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push(tif);
        cv_.notify_one();
    }

    void readSliceRange(Image3D& image, const BoxCoord& box,
                        size_t zStart, size_t zEnd,
                        tsize_t scanlineSize) {
        TIFF* tif = acquire();

        std::vector<char> stripBuf;
        std::vector<float> rowData(box.dimensions.width);

        for (size_t zi = zStart; zi < zEnd; ++zi) {
            uint32_t ifdIndex = static_cast<uint32_t>(box.position.depth + zi);
            if (!TIFFSetDirectory(tif, ifdIndex)) {
                release(tif);
                throw std::runtime_error("Failed to set directory " + std::to_string(ifdIndex));
            }

            uint32_t rowsPerStrip = 0;
            TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
            if (rowsPerStrip == 0) rowsPerStrip = static_cast<uint32_t>(box.dimensions.height);

            uint32_t numStrips = TIFFNumberOfStrips(tif);
            tsize_t actualStripSize = TIFFStripSize(tif);
            if (actualStripSize > static_cast<tsize_t>(stripBuf.size())) {
                stripBuf.resize(actualStripSize);
            }

            for (uint32_t s = 0; s < numStrips; ++s) {
                tmsize_t bytesRead = TIFFReadEncodedStrip(tif, s, stripBuf.data(), actualStripSize);
                if (bytesRead == -1) {
                    release(tif);
                    throw std::runtime_error("Failed to read strip " + std::to_string(s) +
                                             " in z-slice " + std::to_string(ifdIndex));
                }

                uint32_t stripStartRow = s * rowsPerStrip;
                uint32_t rowsInStrip = std::min(rowsPerStrip,
                    static_cast<uint32_t>(box.dimensions.height) - stripStartRow);

                for (uint32_t row = 0; row < rowsInStrip; ++row) {
                    uint32_t y = stripStartRow + row;
                    if (y >= box.dimensions.height) break;
                    const float* scanlineData = reinterpret_cast<const float*>(
                        stripBuf.data() + row * scanlineSize);
                    std::copy_n(scanlineData, box.dimensions.width, rowData.data());
                    image.setRow(y, zi, rowData.data());
                }
            }
        }

        release(tif);
    }

    ImageMetaData metaData_;
    size_t numThreads_;
    std::vector<TIFF*> handles_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<TIFF*> available_;
};

// ---------------------------------------------------------------------------
// Test image generation
// ---------------------------------------------------------------------------
static Image3D makeTestImage(size_t W, size_t H, size_t D) {
    Image3D img(CuboidShape{W, H, D}, 0.0f);
    float denom = static_cast<float>(W * H * D);
    for (size_t z = 0; z < D; ++z) {
        for (size_t y = 0; y < H; ++y) {
            std::vector<float> row(W);
            for (size_t x = 0; x < W; ++x) {
                row[x] = static_cast<float>(z * W * H + y * W + x) / denom;
            }
            img.setRow(y, z, row.data());
        }
    }
    return img;
}

static float expectedPixel(size_t x, size_t y, size_t z, size_t W, size_t H, size_t D) {
    return static_cast<float>(z * W * H + y * W + x) / static_cast<float>(W * H * D);
}

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;

template<typename Fn>
static double timeMs(Fn&& fn) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ---------------------------------------------------------------------------
// Benchmark modes
// ---------------------------------------------------------------------------

struct BenchmarkConfig {
    size_t W = 1048;
    size_t H = 1048;
    size_t D = 1048;
    size_t subDepth = 32;
    int iterations = 5;
    int warmupIterations = 1;
    size_t parallelThreads = 4;
};

// Mode 1: Sync — single thread, two sequential getSubimage calls
static double runSync(const std::string& filename, const BenchmarkConfig& cfg, int channel) {
    size_t memLimit = 4 * cfg.W * cfg.H * cfg.D * sizeof(float);
    ReaderConfig rc{1, memLimit};
    TiffReader reader(filename);
    reader.configure(channel, rc);

    BoxCoord box1{CuboidPosition{0, 0, 0},
                  CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
    BoxCoord box2{CuboidPosition{0, 0, (int64_t)cfg.subDepth},
                  CuboidShape{cfg.W, cfg.H, cfg.subDepth}};

    return timeMs([&] {
        Image3D img1 = reader.getSubimage(box1);
        Image3D img2 = reader.getSubimage(box2);
    });
}

// Mode 2: Async — two threads, two parallel getSubimage calls
static double runAsync(const std::string& filename, const BenchmarkConfig& cfg, int channel) {
    size_t memLimit = 4 * cfg.W * cfg.H * cfg.D * sizeof(float);
    ReaderConfig rc{2, memLimit};
    TiffReader reader(filename);
    reader.configure(channel, rc);

    BoxCoord box1{CuboidPosition{0, 0, 0},
                  CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
    BoxCoord box2{CuboidPosition{0, 0, (int64_t)cfg.subDepth},
                  CuboidShape{cfg.W, cfg.H, cfg.subDepth}};

    return timeMs([&] {
        Image3D img1, img2;
        std::thread t1([&] { img1 = reader.getSubimage(box1); });
        std::thread t2([&] { img2 = reader.getSubimage(box2); });
        t1.join();
        t2.join();
    });
}

// Mode 3: Z-slice parallel — 4 threads, z-slice parallelism per getSubimage
static double runZSliceParallel(const std::string& filename, const ImageMetaData& metaData,
                                const BenchmarkConfig& cfg, int channel) {
    ParallelZSliceReader reader(filename, metaData, cfg.parallelThreads);

    BoxCoord box1{CuboidPosition{0, 0, 0},
                  CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
    BoxCoord box2{CuboidPosition{0, 0, (int64_t)cfg.subDepth},
                  CuboidShape{cfg.W, cfg.H, cfg.subDepth}};

    return timeMs([&] {
        Image3D img1 = reader.getSubimage(box1);
        Image3D img2 = reader.getSubimage(box2);
    });
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------
static bool verifyImage(const Image3D& img, const BoxCoord& box,
                        size_t W, size_t H, size_t D) {
    size_t checks = 0;
    for (size_t z = 0; z < box.dimensions.depth; z += std::max<size_t>(1, box.dimensions.depth / 8)) {
        for (size_t y = 0; y < box.dimensions.height; y += std::max<size_t>(1, box.dimensions.height / 8)) {
            size_t gx = (box.dimensions.width > 1) ? y % box.dimensions.width : 0;
            size_t globalZ = static_cast<size_t>(box.position.depth) + z;
            size_t globalY = static_cast<size_t>(box.position.height) + y;
            float expected = expectedPixel(gx, globalY, globalZ, W, H, D);
            float actual = img.getPixel(gx, y, z);
            if (std::abs(expected - actual) > 1e-5f) {
                std::cerr << "  MISMATCH at (" << gx << "," << y << "," << z
                          << "): expected=" << expected << " actual=" << actual << "\n";
                return false;
            }
            checks++;
        }
    }
    return checks > 0;
}

// ---------------------------------------------------------------------------
// Run a benchmark mode for N iterations
// ---------------------------------------------------------------------------
struct ModeResult {
    double avgMs;
    double minMs;
    double maxMs;
};

template<typename Fn>
static ModeResult runIterations(const std::string& label, int warmup, int iters, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        times.push_back(fn());
    }

    double sum = 0, mn = 1e18, mx = 0;
    for (double t : times) {
        sum += t;
        mn = std::min(mn, t);
        mx = std::max(mx, t);
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  " << label << "\n";
    std::cout << "    avg=" << (sum / iters) << " ms"
              << "  min=" << mn << " ms"
              << "  max=" << mx << " ms\n";

    return {sum / iters, mn, mx};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::err);

    BenchmarkConfig cfg;
    if (argc > 1) cfg.W = std::stoul(argv[1]);
    if (argc > 2) cfg.H = std::stoul(argv[2]);
    if (argc > 3) cfg.D = std::stoul(argv[3]);
    if (argc > 4) cfg.iterations = std::stoi(argv[4]);
    if (argc > 5) cfg.parallelThreads = std::stoul(argv[5]);

    std::string dir = "/tmp/dolphin/tiff_bench";
    fs::create_directories(dir);

    std::string fileUncompressed = dir + "/test_uncompressed.tif";
    std::string fileCompressed   = dir + "/test_lzw.tif";

    double imgMB = static_cast<double>(cfg.W * cfg.H * cfg.D * sizeof(float)) / (1024.0 * 1024.0);
    double subMB = static_cast<double>(cfg.W * cfg.H * cfg.subDepth * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "=== TIFF Parallel Read Benchmark ===\n";
    std::cout << "Image: " << cfg.W << " x " << cfg.H << " x " << cfg.D
              << " (float32, " << std::fixed << std::setprecision(2) << imgMB << " MB)\n";
    std::cout << "Subimage: " << cfg.W << " x " << cfg.H << " x " << cfg.subDepth
              << " (" << subMB << " MB each, 2 reads per iteration)\n";
    std::cout << "Iterations: " << cfg.iterations
              << " (+" << cfg.warmupIterations << " warmup)\n";
    std::cout << "Parallel threads (mode 3): " << cfg.parallelThreads << "\n\n";

    // Generate test image
    std::cout << "Generating test image...\n";
    Image3D testImage = makeTestImage(cfg.W, cfg.H, cfg.D);

    // Write uncompressed
    std::cout << "Writing uncompressed TIFF...\n";
    {
        WriterCompressionConfig comp{COMPRESSION_NONE, -1};
        if (!TiffWriter::writeToFile(fileUncompressed, testImage, comp)) {
            std::cerr << "Failed to write uncompressed TIFF\n";
            return 1;
        }
    }

    // Write LZW-compressed
    std::cout << "Writing LZW-compressed TIFF...\n";
    {
        WriterCompressionConfig comp{COMPRESSION_LZW, -1};
        if (!TiffWriter::writeToFile(fileCompressed, testImage, comp)) {
            std::cerr << "Failed to write compressed TIFF\n";
            return 1;
        }
    }

    double uncompSizeMB = static_cast<double>(fs::file_size(fileUncompressed)) / (1024.0 * 1024.0);
    double compSizeMB = static_cast<double>(fs::file_size(fileCompressed)) / (1024.0 * 1024.0);
    std::cout << "  Uncompressed: " << std::fixed << std::setprecision(2) << uncompSizeMB << " MB\n";
    std::cout << "  Compressed:   " << compSizeMB << " MB"
              << " (ratio: " << (uncompSizeMB / compSizeMB) << ":1)\n\n";

    // Read metadata
    auto metaOptUncomp = TiffReader::readMetadata(fileUncompressed);
    auto metaOptComp = TiffReader::readMetadata(fileCompressed);
    if (!metaOptUncomp || !metaOptComp) {
        std::cerr << "Failed to read metadata\n";
        return 1;
    }

    // Correctness check (one iteration per mode per file)
    std::cout << "Correctness check...\n";
    {
        // Mode 1 on uncompressed
        size_t memLimit = 4 * cfg.W * cfg.H * cfg.D * sizeof(float);
        ReaderConfig rc{1, memLimit};
        TiffReader reader(fileUncompressed);
        reader.configure(0, rc);
        BoxCoord box1{CuboidPosition{0,0,0}, CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
        BoxCoord box2{CuboidPosition{0,0,(int64_t)cfg.subDepth}, CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
        Image3D img1 = reader.getSubimage(box1);
        Image3D img2 = reader.getSubimage(box2);
        if (!verifyImage(img1, box1, cfg.W, cfg.H, cfg.D) ||
            !verifyImage(img2, box2, cfg.W, cfg.H, cfg.D)) {
            std::cerr << "  Mode 1 (sync) FAILED verification!\n";
            return 1;
        }
        std::cout << "  Mode 1 (sync): OK\n";
    }
    {
        // Mode 2 on uncompressed
        size_t memLimit = 4 * cfg.W * cfg.H * cfg.D * sizeof(float);
        ReaderConfig rc{2, memLimit};
        TiffReader reader(fileUncompressed);
        reader.configure(0, rc);
        BoxCoord box1{CuboidPosition{0,0,0}, CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
        BoxCoord box2{CuboidPosition{0,0,(int64_t)cfg.subDepth}, CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
        Image3D img1, img2;
        std::thread t1([&] { img1 = reader.getSubimage(box1); });
        std::thread t2([&] { img2 = reader.getSubimage(box2); });
        t1.join(); t2.join();
        if (!verifyImage(img1, box1, cfg.W, cfg.H, cfg.D) ||
            !verifyImage(img2, box2, cfg.W, cfg.H, cfg.D)) {
            std::cerr << "  Mode 2 (async) FAILED verification!\n";
            return 1;
        }
        std::cout << "  Mode 2 (async): OK\n";
    }
    {
        // Mode 3 on uncompressed
        ParallelZSliceReader reader(fileUncompressed, *metaOptUncomp, cfg.parallelThreads);
        BoxCoord box1{CuboidPosition{0,0,0}, CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
        BoxCoord box2{CuboidPosition{0,0,(int64_t)cfg.subDepth}, CuboidShape{cfg.W, cfg.H, cfg.subDepth}};
        Image3D img1 = reader.getSubimage(box1);
        Image3D img2 = reader.getSubimage(box2);
        if (!verifyImage(img1, box1, cfg.W, cfg.H, cfg.D) ||
            !verifyImage(img2, box2, cfg.W, cfg.H, cfg.D)) {
            std::cerr << "  Mode 3 (z-parallel) FAILED verification!\n";
            return 1;
        }
        std::cout << "  Mode 3 (z-parallel): OK\n";
    }
    std::cout << "\n";

    // ---- Uncompressed benchmarks ----
    std::cout << "--- Uncompressed TIFF (" << uncompSizeMB << " MB) ---\n";
    auto r1u = runIterations("Mode 1: Sync (1 thread, 2 sequential reads)",
        cfg.warmupIterations, cfg.iterations,
        [&] { return runSync(fileUncompressed, cfg, 0); });

    auto r2u = runIterations("Mode 2: Async (2 threads, 2 parallel reads)",
        cfg.warmupIterations, cfg.iterations,
        [&] { return runAsync(fileUncompressed, cfg, 0); });

    auto r3u = runIterations("Mode 3: Z-slice parallel (" + std::to_string(cfg.parallelThreads) + " threads)",
        cfg.warmupIterations, cfg.iterations,
        [&] { return runZSliceParallel(fileUncompressed, *metaOptUncomp, cfg, 0); });

    std::cout << "  Speedup async vs sync:    " << std::fixed << std::setprecision(2)
              << (r1u.avgMs / r2u.avgMs) << "x\n";
    std::cout << "  Speedup z-parallel vs sync: " << (r1u.avgMs / r3u.avgMs) << "x\n\n";

    // ---- Compressed benchmarks ----
    std::cout << "--- LZW-compressed TIFF (" << compSizeMB << " MB) ---\n";
    auto r1c = runIterations("Mode 1: Sync (1 thread, 2 sequential reads)",
        cfg.warmupIterations, cfg.iterations,
        [&] { return runSync(fileCompressed, cfg, 0); });

    auto r2c = runIterations("Mode 2: Async (2 threads, 2 parallel reads)",
        cfg.warmupIterations, cfg.iterations,
        [&] { return runAsync(fileCompressed, cfg, 0); });

    auto r3c = runIterations("Mode 3: Z-slice parallel (" + std::to_string(cfg.parallelThreads) + " threads)",
        cfg.warmupIterations, cfg.iterations,
        [&] { return runZSliceParallel(fileCompressed, *metaOptComp, cfg, 0); });

    std::cout << "  Speedup async vs sync:    " << std::fixed << std::setprecision(2)
              << (r1c.avgMs / r2c.avgMs) << "x\n";
    std::cout << "  Speedup z-parallel vs sync: " << (r1c.avgMs / r3c.avgMs) << "x\n\n";

    std::cout << "Note: Reads are from OS page cache.\n";
    std::cout << "      Uncompressed: bottleneck is memory copy + ITK overhead.\n";
    std::cout << "      Compressed: bottleneck includes LZW decompression (CPU-bound).\n";

    return 0;
}
