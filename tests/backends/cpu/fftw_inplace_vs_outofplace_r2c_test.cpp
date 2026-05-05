/**
 * @file fftw_inplace_vs_outofplace_r2c_test.cpp
 * @brief Benchmark comparing execution time of in-place vs out-of-place FFTW r2c/c2r transforms
 *
 * This benchmark measures and compares the execution time of in-place and
 * out-of-place real-to-complex (and complex-to-real) FFTW transforms.
 *
 * In-place: uses padded real buffer (width = 2*(Nx/2+1)) allocated via
 *           CPUBackendMemoryManager::allocateMemoryOnDeviceRealFFTInPlace,
 *           and the complex output overlays the same buffer.
 *
 * Out-of-place: uses a separate, unpadded real input buffer (width = Nx)
 *               and a separate complex output buffer (width = Nx/2+1).
 *               The plan is created locally in this test using
 *               fftwf_plan_many_dft_r2c with appropriate embed parameters.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>
#include <iomanip>
#include <limits>
#include <chrono>

extern "C" {
#include <fftw3.h>
}

#include "cpu_backend/CPUBackend.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphinbackend/ComplexData.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

typedef float real_t;
typedef real_t complex_t[2];

static constexpr int NUM_ITERATIONS = 10;

/**
 * Generate random real-valued data into a std::vector.
 */
static void generateRandomRealData(std::vector<real_t>& data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

/**
 * Create an out-of-place FFTW r2c plan for a 3D transform.
 *
 * The real input has no padding (logical width = Nx), and the complex output
 * has width Nx/2+1. Two separate buffers are used.
 */
static fftwf_plan createOutOfPlaceR2CPlan(int Nx, int Ny, int Nz) {
    // Allocate temporary buffers for plan creation
    real_t* in  = (real_t*)fftwf_malloc(sizeof(real_t) * Nz * Ny * Nx);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Nz * Ny * (Nx / 2 + 1));

    int rank = 3;
    int n[3] = {Nz, Ny, Nx};
    int inembed[3] = {Nz, Ny, Nx};               // unpadded real input
    int onembed[3] = {Nz, Ny, Nx / 2 + 1};      // complex output (halved)
    int istride = 1, ostride = 1;
    int idist = Nz * Ny * Nx;
    int odist = Nz * Ny * (Nx / 2 + 1);

    fftwf_plan plan = fftwf_plan_many_dft_r2c(
        rank, n, 1,
        in, inembed, istride, idist,
        out, onembed, ostride, odist,
        FFTW_MEASURE
    );

    fftwf_free(in);
    fftwf_free(out);
    return plan;
}

/**
 * Create an out-of-place FFTW c2r plan for a 3D transform.
 */
static fftwf_plan createOutOfPlaceC2RPlan(int Nx, int Ny, int Nz) {
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Nz * Ny * (Nx / 2 + 1));
    real_t* out = (real_t*)fftwf_malloc(sizeof(real_t) * Nz * Ny * Nx);

    int rank = 3;
    int n[3] = {Nz, Ny, Nx};
    int inembed[3] = {Nz, Ny, Nx / 2 + 1};      // complex input (halved)
    int onembed[3] = {Nz, Ny, Nx};               // unpadded real output
    int istride = 1, ostride = 1;
    int idist = Nz * Ny * (Nx / 2 + 1);
    int odist = Nz * Ny * Nx;

    fftwf_plan plan = fftwf_plan_many_dft_c2r(
        rank, n, 1,
        in, inembed, istride, idist,
        out, onembed, ostride, odist,
        FFTW_MEASURE
    );

    fftwf_free(in);
    fftwf_free(out);
    return plan;
}

// just to mix up cache a bit
void square(real_t* in, size_t length){
    for (int i = 0; i < length; i++){
         in[i] = in[i] * in[i];
    }
}

// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== FFTW In-Place vs Out-of-Place R2C Timing Benchmark ===" << std::endl;
    std::cout << "Iterations per measurement: " << NUM_ITERATIONS << std::endl;

    bool allPassed = true;
    fftwf_init_threads();

    fftwf_plan_with_nthreads(8); // each thread that calls the fftw_execute should run the fftw singlethreaded, but its called in parallel

    struct TestConfig {
        int Nx, Ny, Nz;
        const char* label;
    };

    std::vector<TestConfig> configs = {
        {64,  64,  64,  "64x64x64"},
        {128, 128, 64,  "128x128x64"},
        {256, 128, 32,  "256x128x32"},
        {768 , 720 , 120, "768x720x120"},
        {100, 80,  60,  "100x80x60 (non-power-of-2)"},
    };

    using clock = std::chrono::high_resolution_clock;
    using duration_ms = std::chrono::duration<double, std::milli>;

    for (const auto& cfg : configs) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Testing shape: " << cfg.label << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        int Nx = cfg.Nx, Ny = cfg.Ny, Nz = cfg.Nz;
        int volume = Nx * Ny * Nz;
        int complexWidth = Nx / 2 + 1;
        int complexVolume = Nz * Ny * complexWidth;
        int paddedWidth = 2 * (Nx / 2 + 1);

        // ---- Generate input data ----
        std::vector<real_t> hostInput;
        generateRandomRealData(hostInput, volume);

        // ================================================================
        // IN-PLACE R2C + C2R
        // ================================================================
        std::cout << "Planning inplace" << std::endl;
        real_t* inplace_real = (real_t*)fftwf_malloc(sizeof(real_t) * Nz * Ny * paddedWidth);
        std::memset(inplace_real, 0, sizeof(real_t) * Nz * Ny * paddedWidth);

        // Copy input into padded buffer
        for (int z = 0; z < Nz; ++z)
            for (int y = 0; y < Ny; ++y)
                for (int x = 0; x < Nx; ++x)
                    inplace_real[z * Ny * paddedWidth + y * paddedWidth + x] =
                        hostInput[z * Ny * Nx + y * Nx + x];

        // In-place r2c plan
        int n_ip[3]       = {Nz, Ny, Nx};
        int inembed_ip[3] = {Nz, Ny, paddedWidth};
        int onembed_ip[3] = {Nz, Ny, complexWidth};

        fftwf_plan inplace_r2c = fftwf_plan_many_dft_r2c(
            3, n_ip, 1,
            inplace_real, inembed_ip, 1, Nz * Ny * paddedWidth,
            (fftwf_complex*)inplace_real, onembed_ip, 1, Nz * Ny * complexWidth,
            FFTW_MEASURE
        );
        assert(inplace_r2c != nullptr && "In-place r2c plan creation failed");

        // In-place c2r plan
        fftwf_plan inplace_c2r = fftwf_plan_many_dft_c2r(
            3, n_ip, 1,
            (fftwf_complex*)inplace_real, onembed_ip, 1, Nz * Ny * complexWidth,
            inplace_real, inembed_ip, 1, Nz * Ny * paddedWidth,
            FFTW_MEASURE
        );
        assert(inplace_c2r != nullptr && "In-place c2r plan creation failed");

        // ---- Time in-place r2c ----
        // Reset input before timing
        std::memset(inplace_real, 0, sizeof(real_t) * Nz * Ny * paddedWidth);
        for (int z = 0; z < Nz; ++z)
            for (int y = 0; y < Ny; ++y)
                for (int x = 0; x < Nx; ++x)
                    inplace_real[z * Ny * paddedWidth + y * paddedWidth + x] =
                        hostInput[z * Ny * Nx + y * Nx + x];

        std::cout << "Executing in place" << std::endl;
        auto ip_r2c_start = clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            fftwf_execute_dft_r2c(inplace_r2c, inplace_real, (fftwf_complex*)inplace_real);
            square(inplace_real, volume);
            fftwf_execute_dft_c2r(inplace_c2r, (fftwf_complex*)inplace_real, inplace_real);
        }
        auto ip_r2c_end = clock::now();
        double ip_r2c_ms = duration_ms(ip_r2c_end - ip_r2c_start).count() / NUM_ITERATIONS;


        double ip_total_ms = ip_r2c_ms;

        fftwf_destroy_plan(inplace_r2c);
        fftwf_destroy_plan(inplace_c2r);
        fftwf_free(inplace_real);

        // ================================================================
        // OUT-OF-PLACE R2C + C2R
        // ================================================================
        std::cout << "Planning outofplace" << std::endl;
        real_t* oop_real_in  = (real_t*)fftwf_malloc(sizeof(real_t) * volume);
        fftwf_complex* oop_complex = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * complexVolume);
        real_t* oop_real_out = (real_t*)fftwf_malloc(sizeof(real_t) * volume);
        std::memcpy(oop_real_in, hostInput.data(), sizeof(real_t) * volume);

        fftwf_plan oop_r2c = createOutOfPlaceR2CPlan(Nx, Ny, Nz);
        assert(oop_r2c != nullptr && "Out-of-place r2c plan creation failed");

        fftwf_plan oop_c2r = createOutOfPlaceC2RPlan(Nx, Ny, Nz);
        assert(oop_c2r != nullptr && "Out-of-place c2r plan creation failed");

        std::cout << "Executing outofplace" << std::endl;
        // ---- Time out-of-place r2c ----
        auto oop_r2c_start = clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            fftwf_execute_dft_r2c(oop_r2c, oop_real_in, oop_complex);
            square(oop_real_in, volume);
            fftwf_execute_dft_c2r(oop_c2r, oop_complex, oop_real_out);
        }
        auto oop_r2c_end = clock::now();
        double oop_r2c_ms = duration_ms(oop_r2c_end - oop_r2c_start).count() / NUM_ITERATIONS;


        double oop_total_ms = oop_r2c_ms;

        fftwf_destroy_plan(oop_r2c);
        fftwf_destroy_plan(oop_c2r);
        fftwf_free(oop_real_in);
        fftwf_free(oop_complex);
        fftwf_free(oop_real_out);

        // ================================================================
        // TIMING RESULTS
        // ================================================================
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n  --- Timing Results (avg over " << NUM_ITERATIONS << " iterations) ---" << std::endl;
        std::cout << "                        In-Place      Out-of-Place    Ratio (IP/OOP)" << std::endl;
        std::cout << "    R2C:           "
                  << std::setw(10) << ip_r2c_ms << " ms  "
                  << std::setw(10) << oop_r2c_ms << " ms    "
                  << std::setw(6) << std::setprecision(3) << (ip_r2c_ms / oop_r2c_ms) << std::endl;
        std::cout << "    R2C + C2R:     "
                  << std::setw(10) << ip_total_ms << " ms  "
                  << std::setw(10) << oop_total_ms << " ms    "
                  << std::setw(6) << std::setprecision(3) << (ip_total_ms / oop_total_ms) << std::endl;

        std::string faster = (ip_total_ms < oop_total_ms) ? "in-place" : "out-of-place";
        double speedup = (ip_total_ms < oop_total_ms)
            ? (oop_total_ms / ip_total_ms) : (ip_total_ms / oop_total_ms);
        std::cout << "    --> " << faster << " is faster by "
                  << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }

    // ================================================================
    // Additional benchmark: In-place via CPUBackendMemoryManager
    // ================================================================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Benchmarking in-place via CPUBackendMemoryManager" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    {
        int Nx = 128, Ny = 64, Nz = 32;
        int volume = Nx * Ny * Nz;
        int paddedWidth = 2 * (Nx / 2 + 1);
        int complexWidth = Nx / 2 + 1;
        int complexVolume = Nz * Ny * complexWidth;

        CuboidShape shape(Nx, Ny, Nz);
        std::string backendName = "default";

        IBackendManager& manager = BackendFactory::getInstance().getBackendManager(backendName);
        BackendConfig config;
        config.nThreads = 8;
        IBackend& backend = manager.getBackend(config);
        IDeconvolutionBackend& deconvBackend = backend.mutableDeconvManager();
        IBackendMemoryManager& memManager = backend.mutableMemoryManager();
        // Use the CPUBackendManager's allocation method for in-place r2c
        RealData realData = memManager.allocateMemoryOnDeviceRealFFTInPlace(shape);

        // Generate input and fill padded buffer
        std::vector<real_t> hostInput;
        generateRandomRealData(hostInput, volume);

        std::memset(realData.getData(), 0, sizeof(real_t) * Nz * Ny * paddedWidth);
        for (int z = 0; z < Nz; ++z)
            for (int y = 0; y < Ny; ++y)
                for (int x = 0; x < Nx; ++x)
                    realData.getData()[z * Ny * paddedWidth + y * paddedWidth + x] =
                        hostInput[z * Ny * Nx + y * Nx + x];

        // Reinterpret as complex (in-place view)
        ComplexData complexData = memManager.reinterpret(realData);

        // ---- Time in-place r2c via CPUBackend ----
        auto bk_r2c_start = clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            deconvBackend.forwardFFT(realData, complexData);
            square(realData.getData(), complexVolume);
            deconvBackend.backwardFFT(complexData, realData);
        }
        auto bk_r2c_end = clock::now();
        double bk_r2c_ms = duration_ms(bk_r2c_end - bk_r2c_start).count() / NUM_ITERATIONS;

        double bk_total_ms = bk_r2c_ms ;

        // Out-of-place for comparison
        real_t* oop_in = (real_t*)fftwf_malloc(sizeof(real_t) * volume);
        fftwf_complex* oop_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * complexVolume);
        real_t* oop_real_out = (real_t*)fftwf_malloc(sizeof(real_t) * volume);
        std::memcpy(oop_in, hostInput.data(), sizeof(real_t) * volume);

        fftwf_plan oop_r2c = createOutOfPlaceR2CPlan(Nx, Ny, Nz);
        assert(oop_r2c != nullptr);
        fftwf_plan oop_c2r = createOutOfPlaceC2RPlan(Nx, Ny, Nz);
        assert(oop_c2r != nullptr);

        // ---- Time out-of-place r2c ----
        auto oop_r2c_start = clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            fftwf_execute_dft_r2c(oop_r2c, oop_in, oop_out);
        }
        auto oop_r2c_end = clock::now();
        double oop_r2c_ms = duration_ms(oop_r2c_end - oop_r2c_start).count() / NUM_ITERATIONS;

        // ---- Time out-of-place c2r ----
        auto oop_c2r_start = clock::now();
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            fftwf_execute_dft_c2r(oop_c2r, oop_out, oop_real_out);
        }
        auto oop_c2r_end = clock::now();
        double oop_c2r_ms = duration_ms(oop_c2r_end - oop_c2r_start).count() / NUM_ITERATIONS;

        double oop_total_ms = oop_r2c_ms + oop_c2r_ms;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n  --- Timing Results (avg over " << NUM_ITERATIONS << " iterations) ---" << std::endl;
        std::cout << "                        In-Place (BK)  Out-of-Place    Ratio (IP/OOP)" << std::endl;
        std::cout << "    R2C:           "
                  << std::setw(10) << bk_r2c_ms << " ms  "
                  << std::setw(10) << oop_r2c_ms << " ms    "
                  << std::setw(6) << std::setprecision(3) << (bk_r2c_ms / oop_r2c_ms) << std::endl;
        std::cout << "    R2C + C2R:     "
                  << std::setw(10) << bk_total_ms << " ms  "
                  << std::setw(10) << oop_total_ms << " ms    "
                  << std::setw(6) << std::setprecision(3) << (bk_total_ms / oop_total_ms) << std::endl;

        std::string faster = (bk_total_ms < oop_total_ms) ? "in-place (BK)" : "out-of-place";
        double speedup = (bk_total_ms < oop_total_ms)
            ? (oop_total_ms / bk_total_ms) : (bk_total_ms / oop_total_ms);
        std::cout << "    --> " << faster << " is faster by "
                  << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

        fftwf_destroy_plan(oop_r2c);
        fftwf_destroy_plan(oop_c2r);
        fftwf_free(oop_in);
        fftwf_free(oop_out);
        fftwf_free(oop_real_out);
    }

    // ================================================================
    // Final summary
    // ================================================================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (allPassed) {
        std::cout << "=== BENCHMARK COMPLETED ===" << std::endl;
        return 0;
    } else {
        std::cout << "=== BENCHMARK HAD ISSUES ===" << std::endl;
        return 1;
    }
}

