# AGENTS.md

Guidance for coding agents working in this repository. Everything below was verified
against the tree at the time of writing. When code and this file disagree, **the code
wins** — then please fix this file.

## Project Overview

DOLPHIN (Deconvolution with Optimized Local PSFs for High-speed Image recoNstruction) is a
C++20 scientific computing application for 3D microscopy image deconvolution. It ships a
CLI frontend; the GUI frontend exists but is out of the main build (see [Frontends](#frontends)).

Pipeline in one line: load image (TIFF) → build/load PSF → pad → split into cubes → pad each
cube → copy to backend → FFT-based deconvolution per cube on worker threads → stitch cubes back →
write TIFF.

## Repository Map

| Path | Contents |
|---|---|
| `src/`, `include/dolphin/` | Core library (`dolphin` static lib). Facade, configs, services, strategies, algorithms, PSF handling, threads, logging. |
| `dolphin_image/` | In-tree static lib (`dolphin_image`): `Image3D` (wraps `itk::Image<float,3>`), TIFF reader/writer, padding, image ops. **Not** a submodule. |
| `backends/dolphinbackend/` | Backend-agnostic layer: `IBackend`, `IBackendManager`, `IComputeBackend`, `IBackendMemoryManager`, `ManagedData`, `MemoryFootprint`, `ThreadConfig`. |
| `backends/cpu/` | FFTW3f implementation, FFTW wisdom cache. |
| `backends/cuda/` | cuFFT implementation, device memory mgmt; `backends/cuda/lib/cube/` = vendored CUBE kernel library. |
| `frontends/cli/` | CLI frontend, built by the root build (`BUILD_CLI`). |
| `frontends/gui/` | Standalone CMake project (Dear ImGui + GLFW). Not in root build, not in CI, currently API-broken. |
| `tests/` | GoogleTest suite, one combined binary. |
| `benchmarks/` | Standalone benchmark executables (not Google Benchmark). |
| `configs_template/` | **Committed** example JSON configs — the reference for config shape. |
| `configs/` | **Git-ignored** local user configs (personal paths). Never commit; copy from `configs_template/`. |
| `cmake/` | `DolphinPCH.cmake` (precompiled headers), `dolphinConfig.cmake.in`, `vcpkg/` (Windows manifest + ITK overlay port). |
| `.github/` | CI workflows, Docker build files, Brewfile. |
| `lib/` | Git submodules: `nlohmann/json`, `spdlog`. |
| `frontends/cli/lib/`, `frontends/gui/lib/` | Submodules: CLI11, imgui, imgui-filebrowser, implot3d. |
| `docs/` | Poisson-MLE algorithm walkthrough + math derivations. |
| `scripts/`, `profiling/` | Docker build helper, ORAS release publisher; profiler wrapper scripts. |

Notable files: `include/dolphin/Dolphin.h` (facade), `include/dolphin/Config.h` (config system),
`include/dolphin/deconvolution/AlgorithmNames.h` (algorithm name constants),
`src/deconvolution/DeconvolutionAlgorithmFactory.cpp` (registration),
`include/dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h` (algorithm base class),
`src/deconvolution/deconvolutionStrategies/` (planning, splitting, execution, PSF handling).

## Build

Submodules are required: `git submodule update --init --recursive`.

```bash
cmake --preset dev            # Debug, native CUDA arch, tests ON, benchmarks OFF → build/
cmake --build --preset dev
ctest --preset dev
```

| Preset | Purpose |
|---|---|
| `dev` | Default developer build: `Debug`, `CMAKE_CUDA_ARCHITECTURES=native`, `ENABLE_TESTS=ON`, `ENABLE_BENCHMARKS=OFF`, `DOLPHIN_INDIVIDUAL_TESTS=OFF` |
| `dev-ccache`, `dev-ninja`, `dev-ninja-ccache` | Same, with ccache and/or Ninja (tools must be installed) |
| `release` | `Release`, archs `75;80;90`, benchmarks ON, `BUILD_CLI=ON` → `build-release/` |
| `cpu-only` | `Release`, `BUILD_CUDA=OFF` → `build-cpu/` |

Without presets: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` (default build type is Release).
CMake ≥ 3.28 for presets; the project itself only requires 3.18.

### Build options (real defaults)

| Option | Default | Notes |
|---|---|---|
| `BUILD_CUDA` | `ON` | If `CUDAToolkit` 12.8 is **missing**, the CUDA backend silently drops out (configure still succeeds). |
| `BUILD_CLI` | `ON` | Builds `dolphin_cli` → `build/frontends/cli/dolphin`. |
| `ENABLE_TESTS` | `ON` | Combined test binary only. |
| `ENABLE_BENCHMARKS` | `OFF` | |
| `DOLPHIN_INDIVIDUAL_TESTS` | `OFF` | Per-file test executables on top of the combined binary. |
| `BUILD_DOLPHIN_LIBRARY` | `ON` | Set OFF to consume DOLPHIN as a subproject. |
| `DOLPHIN_ENABLE_PCH` | `ON` | Precompiles nlohmann/json, spdlog, `itkImage`. |
| `DOUBLE_PRECISION` | `OFF` | CUBE target only. |

### Dependencies

- **ITK** (built in CI as 5.4.5, static, `ITKIOTIFF` deliberately disabled — TIFF IO goes through LibTIFF directly)
- **LibTIFF** (CI: 4.7.0 static), **FFTW3f** (`--enable-float --enable-threads`; `FFTW_PATH` cache var to override; FFTW wisdom cached at `FFTW_WISDOM_PATH`)
- **CUDA Toolkit ≥ 12.8** (optional), **OpenMP** (optional; required only for FFTW-omp or benchmarks)
- GoogleTest 1.14 via `FetchContent` when tests are on; LIKWID optional for benchmarks
- Vendored submodules: nlohmann/json, spdlog, CLI11 (CLI only)

OpenCV was removed; do not reintroduce it. Image handling is ITK + LibTIFF.

### Compile flags

Per-config flags live in the `project_options` INTERFACE target in the root `CMakeLists.txt`
(not in `CMAKE_CXX_FLAGS`):

- GNU/Clang Release: `-O3 -ffast-math -fno-finite-math-only -DNDEBUG -funroll-loops`
- `-mavx2 -mfma`: **GNU only, x86 hosts only, Release only**
- MSVC: `/O2 /fp:fast`, `/utf-8`, `/Zc:preprocessor`
- Test targets add `-fno-fast-math` for numeric accuracy. **Do not "fix" this asymmetry** —
  the library is fast-math, so never rely on NaN/inf propagation or exact FP semantics.
- Binary portability: `-static-libstdc++ -static-libgcc` linked PUBLIC on `dolphinbackend`
  (GNU, non-Apple)

### Tests

```bash
ctest --preset dev                          # all tests
ctest --preset dev -LE exhaustive           # skip the exhaustive split contract test
ctest --preset dev -E "[Cc][Uu][Dd][Aa]"    # GPU-less runs (CI uses this)
```

- One combined binary `dolphin_all_tests` compiles every source in `ALL_TEST_SOURCES`
  (`tests/CMakeLists.txt`), plus backend and CLI tests conditionally. ~629 ctest cases.
- **Adding a test file takes two edits**: append the `.cpp` to `ALL_TEST_SOURCES` in
  `tests/CMakeLists.txt`, *and* add a `dolphin_add_test(...)` call in the matching subdirectory
  `CMakeLists.txt` (only used with `DOLPHIN_INDIVIDUAL_TESTS=ON`).
- Test targets get `-fno-fast-math` and `DOLPHIN_TEST_RAW_ACCESS` (promotes `ManagedData`
  raw accessors). Test helpers live in `tests/utils/TestUtils.h`; no binary fixtures in the repo —
  images are generated in code and written to `/tmp/dolphin`.
- Test names containing "Cuda" are GPU-runtime tests skipped by CI.

### Benchmarks

`ENABLE_BENCHMARKS=ON` (the `release` preset) builds FFTW scaling benchmarks, TIFF parallel-read
benchmarks and `cufft_vs_fftw_benchmark` into `build/benchmarks/{cpu,cuda}`. See the READMEs there.

## Architecture

### Layer stack

```
frontends/cli (CLI11)  →  IFrontend  →  Dolphin (facade)
                                         ├─ PSFGenerationService
                                         └─ DeconvolutionService
                                              ├─ DeconvolutionStrategy  (createPlan)
                                              └─ DeconvolutionExecutor (execute)
                                                   ├─ DeconvolutionAlgorithm  (per cube)
                                                   ├─ PSFHandler / PSFPreprocessor
                                                   ├─ ImageSplit / padding / Postprocessor
                                                   └─ IComputeBackend + IBackendMemoryManager
                                                        ├─ CPUBackend  (FFTW3f + OpenMP)
                                                        └─ CUDABackend (cuFFT + CUBE)
```

Requests/results are plain structs in `include/dolphin/ServiceAbstractions.h`
(`PSFGenerationRequest`, `DeconvolutionRequest`, `Result<T>`, progress/log callbacks).
Both `Dolphin` methods are **synchronous**; there is no async API.

### Deconvolution pipeline (`src/DeconvolutionService.cpp`)

1. Validate algorithm name, `ThreadConfig`, memory settings.
2. Pick strategy + executor (STANDARD vs LABELED, based on `SetupConfig::getDeconvType()`).
3. Open `TiffReader` (file or directory auto-detected) and `TiffWriter` (path from `setup_config.output`).
4. Create a `DeconvolutionPlan`: resolve `MemoryBudget` (auto = ~90 % of available host/device
   memory), assemble `MemoryFootprint`, generate/load PSFs in the thread pool, compute the padding
   scheme, split the image into cubes (`splitImageHomogeneous`, hard cap **100 cubes**, smooth
   FFT-friendly sizes), create one `CubeTaskDescriptor` per cube plus a per-device `TaskContext`.
5. Optionally save PSFs next to the output.
6. Execute: per cube on an io thread — read unpadded cube, pad into an FFT-layout host buffer,
   copy to device, enqueue to a worker thread. The worker runs the algorithm on device data
   (PSFs preprocessed once per shape/PSF/device: convert → copy → octant shift → forward FFT →
   normalize). Results are copied back and the writer stitches cubes (with feathering/weights
   where configured).
7. Task exceptions are collected; all tasks are awaited; the first error is rethrown and the
   output is not overwritten on failure.

### Backends

- Selection string is `"cpu"` (`HOST_BACKEND`) or `"cuda"`; unknown names fall back to `cpu`
  with a warning.
- `BackendFactory` registers managers at first use; `#if ENABLE_CUDA` guards (the macro is set
  PUBLIC by CMake) hide CUDA registration in CPU-only builds.
- CPU and CUDA paths must produce matching FFT layouts (real-to-complex with `width/2+1`
  rows); `tests/unit/test_cpu_cuda_fftparity.cpp` enforces this. Keep them in sync.
- FFTW plans are cached per (shape, direction, transform type, in-place, OMP threads) and cuFFT
  similarly — prefer few distinct cube shapes, and remember planning cost is real.

### Memory model

- `ManagedData<T>` (`backends/dolphinbackend/include/dolphinbackend/ComplexData.h`) is an RAII
  buffer with owner backend, size, row padding. Aliases: `complex_t`, `real_t`, `ComplexData`,
  `RealData`, `ComplexView`/`RealView` (non-owning).
- Raw element access is `protected`; only `IBackendMemoryManager` and `IComputeBackend` may touch
  device memory. Element accessors validate host reachability and will fail for device buffers.
  **Do not reach around this** — pass device data to the backend, not into loops.
- Buffers are allocated for FFT-in-place layouts: `getDataBytes() != volume * sizeof(T)` because
  rows are padded. Never assume dense contiguous data.
- Every memory-owning subsystem (reader cache, writer reserve, io staging, algorithm working set,
  PSF cache, FFT workspace, padding peak) implements a `MemoryComponent` and reports into the
  `MemoryFootprint`; the plan budgets against `MemoryBudget`. Exceeding caps throws
  `dolphin::backend::MemoryException`.

### Threading

- `ThreadConfig` (`backends/dolphinbackend/include/dolphinbackend/ThreadConfig.h`) carries user
  intent (`n_threads` *or* the pool sizes `n_worker_threads`, `n_io_threads`, `n_reader_threads`);
  mixing them is rejected. The backend manager resolves the actual distribution.
- Custom `ThreadPool` (`include/dolphin/ThreadPool.h`) with a bounded queue and backpressure via
  `setCondition`, used for io, worker and PSF-generation tasks. Reader-side prefetch uses
  `dolphin_image/include/dolphin_image/IO/ReaderThreadPool.h`.
- OpenMP is used **only inside the CPU backend** (FFTW threading).
- `Dolphin::init()` forces `itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1)`.
  This is intentional — ITK threading is replaced by the pools. Do not revert.

### Logging

`include/dolphin/Logging.h`, spdlog, async. `Logging::init(logDir)` is called from
`Dolphin::init`. Named loggers: `backend`, `config`, `psf`, `deconvolution`, `reader`, `writer`,
`default`. Sinks: color console (warn+), `<logDir>/debug.log` (trace), and a frontend sink
forwarding to a `Logging::LogCallback`.

Rules: always `spdlog::get("<name>")` and handle `nullptr` (uninitialized logger) in library code;
backend code logs through the `LogCallback` bridge in `BackendFactory.h`, not spdlog directly.

## Configuration System

Everything lives in `include/dolphin/Config.h` and `src/*Config.cpp`.

- A config class derives from `Config`, holds plain members, and registers every member in
  `registerAllParameters()` as a `ConfigParameter` (type, `void*` value, name, `jsonTag`,
  `cliFlag`, description, range, selection…). JSON load/save, CLI option generation, validation
  and printing all walk that registry via `visitParams()`.
- **`Config` copy/move is deleted** because the registry holds `void*` into members. Every derived
  class must hand-write copy/move that copies fields and then calls `registerAllParameters()`
  again (see `src/SetupConfig.cpp`, `src/deconvolution/DeconvolutionConfig.cpp`). Do not copy the
  `parameters` vector directly — the pointers would dangle.
- `Config::loadJSONFile()` opens the path **as given**; there is no rewriting relative to the
  executable or build dir. Relative paths resolve against the current working directory.
- Unknown JSON keys are logged as warnings; parameters outside their declared range only warn
  (`logParameterRangeViolations`).
- Config classes: `SetupConfigPSF` → `SetupConfig` (adds `image_path`, `psf_file_paths`,
  `labeled_image`, `label_psf_map`, `save_psf`, `psfConfigs`); `DeconvolutionConfig` (base, with
  one derived config per algorithm); PSF configs in `src/psf/configs/` (plus per-modality:
  widefield/confocal); `ThreadConfig` in the backend layer.
- JSON bundle format (see `configs_template/`): three top-level sections `setup_config`,
  `deconvolution_config`, `psf_configs` (array or single object). Split files via `-s/-d/-p`.
- **`Config` file values override CLI flags** for overlapping parameters. Adding a parameter
  needs only the registry entry — the CLI picks it up automatically.
- `configs_template/` holds working example configs; the top-level `README.md` documents the
  JSON format.

## Deconvolution Algorithms

Name constants: `include/dolphin/deconvolution/AlgorithmNames.h` (`namespace DeconvolutionAlgorithms`).
`names()` feeds both config validation and CLI `--algorithm_name`.

| Name string | Algorithm class | Config class |
|---|---|---|
| `RichardsonLucy` | `RLDeconvolutionAlgorithm` | `RichardsonLucyConfig` |
| `Convolution` | `ConvolutionAlgorithm` | `ConvolutionConfig` |
| `InverseFilter` | `InverseFilterDeconvolutionAlgorithm` | `InverseFilterConfig` |
| `RichardsonLucyTotalVariation` | `RLTVDeconvolutionAlgorithm` | `RichardsonLucyTVConfig` |
| `RegularizedInverseFilter` | `RegularizedInverseFilterDeconvolutionAlgorithm` | `RegularizedInverseFilterConfig` |
| `RichardsonLucywithAdaptiveDamping` | `RLADDeconvolutionAlgorithm` | `RichardsonLucyADConfig` |
| `PoissonMLE` | `PoissonMLEUnitStepDeconvolutionAlgorithm` | `PoissonMLEUnitStepConfig` |
| `PoissonMLEAdaptiveStep` | `PoissonMLEDeconvolutionAlgorithm` | `PoissonMLEConfig` |

**Naming trap:** `"PoissonMLE"` runs the **UnitStep** class; the class carrying the plain
`PoissonMLE` name is registered as `"PoissonMLEAdaptiveStep"` (Newton line search). This is
deliberate — see the comment in `DeconvolutionAlgorithmFactory::registerAlgorithms()` and
`docs/poisson-mle-deconvolution*.md`.

### Algorithm contract

`DeconvolutionAlgorithm` (`include/dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h`):

- `configure(const DeconvolutionConfig&)` — base overload downcasts via `asConfig<Derived>(…)`
  (matches on the config's `kAlgorithmName`).
- `init(const CuboidShape&)` — allocate scratch for one cube shape.
- `deconvolve(const ComplexData& H, RealData& g, RealData& f, const DeconvolutionRunContext&)` —
  H/g/f are **already on the backend device**. All arithmetic goes through `IComputeBackend`.
- `getWorkingSetBytes(shape)` — feeds the memory footprint; keep it honest.
- `cloneSpecific()` — per-thread clone.

`DeconvolutionRunContext` carries run-wide values resolved once (e.g. background) so cubes agree.

### Adding an algorithm

1. Add `kName` + entry in `names()` in `include/dolphin/deconvolution/AlgorithmNames.h`.
2. Add a config class in `include/dolphin/deconvolution/configs/` + `src/deconvolution/configs/`:
   `static constexpr const char* kAlgorithmName`, ctor/copy/move calling
   `assignBase(...)`/`registerAllParameters()`.
3. Add the algorithm class in `include/dolphin/deconvolution/algorithms/` + `src/.../algorithms/`
   implementing `configure`, `init`, `deconvolve`, `getWorkingSetBytes`, `cloneSpecific`.
4. Register both creators in `DeconvolutionAlgorithmFactory::registerAlgorithms()`.
5. Add both `.cpp` files to `DOLPHIN_SOURCES` in the root `CMakeLists.txt` (no globbing).
6. Add tests (`tests/unit/test_algorithm_factory.cpp`, `tests/unit/test_algorithms.cpp`) and a
   `configs_template/` example; update the README algorithm list.

The CLI needs no changes — options are generated from the config registry.

## PSF System and Padding

- `PSFCreator` (`include/dolphin/PSFCreator.h`, `src/PSFCreator.cpp`) generates PSFs from configs,
  loads them from TIFF files/directories, and overrides shapes.
- `PSFGeneratorFactory` registers `"Gaussian"` (analytic) and `"GibsonLanni"` (physical model with
  a modality layer). PSFs loaded from files carry no modality metadata.
- Modality: `include/dolphin/psf/configs/ModalityConfig.h` — `WIDEFIELD` (default) or `CONFOCAL`.
  `selectModality()` must run **before** parameters are loaded. Confocal image formation exists
  only in Gibson-Lanni (excitation × detection integral, optional finite pinhole). Mixing
  modalities across PSFs warns; a confocal config without `excitation_wavelength_nm` warns that
  the PSF degrades toward widefield.
- PSF-to-region assignment is either **global** (multiple PSFs each deconvolve the whole image and
  results are combined) or **labeled** (`labeled_image` + `label_psf_map`, ranges via `RangeMap`,
  executed by `LabeledDeconvolutionExecutor`). The old per-subimage/layer PSF arrays are gone.
- Padding resolution order in `PSFHandler::getPadding`:
  `padding_relative_max` > `cube_padding` > preset `padding_strategy`
  (`NONE/LOW/MEDIUM/MAX`; `DEFAULT` → `MEDIUM`). Presets are fractions of the PSF max
  (MEDIUM 1e-4, LOW 1e-2, MAX 1e-20); the generation cutoff is 100× below the trim threshold.
- Result is a `PaddingScheme {insidePadding, imagePadding}` — boundary cube faces get exactly
  `imagePadding`, interior faces `insidePadding`. `feathering_radius > 0` requires padding ≥ radius.
- Fill types (`PaddingFillType`): `ZERO`, `MIRROR`, `LINEAR`, `QUADRATIC`, `SINUSOID`, `GAUSSIAN`.
  For `ZERO` the cube is padded straight into the FFT-layout host buffer (no intermediate image).
- `cube_size` (all zeros = auto) forces fixed cube sizes; `slice_all_dimensions` is the retry
  fallback when Z-only splitting fails.

## Frontends

- **CLI** (`frontends/cli/`): CLI11, subcommands `psfgenerator` and `deconvolution`. Before parsing,
  `prescanAlgorithmName()` peeks at argv so the option tree binds to the right algorithm config
  class; a config file's `algorithm_name` wins over `--algorithm_name`. Options are auto-generated
  from `Config::visitParams`; CLI group names must not contain spaces (CLI11 rule).
- **GUI** (`frontends/gui/`): separate CMake project (`find_package(dolphin REQUIRED)`, Dear ImGui +
  GLFW + GLEW + implot3d). Not built by root presets, not in CI. It currently calls a
  non-existent `Dolphin::deconvolveAsync` and an outdated `PSFGenerationRequest` constructor, so
  **it does not compile**. Treat GUI work as a separate task; never assume it builds.
- `main()` for the CLI lives in `frontends/cli/src/main.cpp`; it calls `dolphin->init(logDir)` then
  `CLIFrontend(...).run()`.

## CI/CD

- `.github/workflows/ci-main.yml` — push/PR to `main` (markdown-only changes ignored):
  `ubuntu-cpu` and `ubuntu-cuda` (Docker images in `.github/docker/`, CUDA image on CUDA 13,
  `ctest -E "[Cc][Uu][Dd][Aa]"`), `macos` (Homebrew bundle, source-built ITK 5.4.5, no CUDA),
  `windows` (MSVC + vcpkg manifest from `cmake/vcpkg/`), `windows-cuda` (CUDA 12.8.1, Ninja, no
  GPU runtime tests).
- `.github/workflows/release.yml` — on `v*` tags: per-platform build → clean-VM smoke test
  (`dolphin --help`) → GitHub Release artifacts → ORAS publish to GHCR
  (`ghcr.io/lennartkoehler/dolphin`) via `scripts/oras_publish_v2.py`. These are OCI bundles of
  binaries, not container images.
- Local Docker build: `scripts/docker-build.sh` (uses `.github/docker/Dockerfile --target builder`).
- CI relies on `-submodules recursive` checkouts and on gtest discovery working inside the
  container (`-DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST`).

## Conventions

- Headers in `include/`, implementations in the mirrored `src/` subtree; namespaced includes
  (`"dolphin/..."`, `"dolphin_image/..."`, `"dolphinbackend/..."`, `"cpu_backend/..."`).
- **Every new `.cpp` must be added to the target's source list** in CMake — there is no globbing.
- Use `real_t`, `complex_t`, `ComplexData`, `RealData` aliases instead of raw types.
- Naming is mixed and project-specific: `n*` for sizes/counts, `k*` for constants, `*Config`
  suffix for config classes, UPPERCASE unscoped enum values, a few Hungarian-style field names in
  structs. Match the surrounding file rather than modernizing names.
- No `.clang-format` or `.editorconfig` — formatting is not enforced, but match neighbours.
- Factories are singletons (`getInstance()`); registration happens in an explicit
  `registerX()` method called lazily on first use.
- Fast iteration levers: `dev-ninja-ccache` preset, PCH (on by default; with ccache set
  `sloppiness=pch_defines,time_macros`), single combined test binary.

## Gotchas — Do Not Fix Blindly

- `-ffast-math` in Release vs `-fno-fast-math` in tests: intentional.
- ITK global thread count forced to 1 in `Dolphin::init`: intentional.
- 100-cube hard limit in `ImageSplit` throws `DolphinImageSplitException`; overflow of the last cube
  per dimension goes into padding-before to keep cubes contiguous.
- `PSFHandler::getPSFs(shape)` **mutates** the PSFs (reshapes/zero-fills) despite looking like a
  getter.
- `ManagedData` element access on device buffers fails by design; tests opt out with
  `DOLPHIN_TEST_RAW_ACCESS` (documented ODR caveat).
- `PSF/Image3D` copy constructor is used heavily for subimages — think about memory when adding copies.
- FFT-in-place buffers are row-padded; `getDataBytes()` is the only safe size.
- Docs drift happens; the config registry in `src/*Config.cpp` is the source of truth for every
  parameter (JSON tag, CLI flag, range, default). `./dolphin deconvolution --help` prints the
  auto-generated option list.
- `TODO.md` and git-tracked-but-deleted `NOTES.md`/`PLAN.md` are the maintainer's working notes;
  don't treat them as specs.
- `.clangd` contains stale paths from another project; fix locally rather than committing guessing
  includes.