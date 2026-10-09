Part 1 — Standard C++ compile-speed techniques
Tooling
Technique	Mechanism
ccache / sccache	CMAKE_C(XX)_COMPILER_LAUNCHER=ccache. Reuses object files across rebuilds/branch switches. Biggest single win in iterative work.
Ninja generator	-G Ninja. Lower build-graph overhead than Make, better parallel scheduling.
Parallel build	cmake --build build -j or Ninja. Serial make wastes cores.
mold / lld linker	Link time cut 2–10x for static-heavy projects.
Build configuration
- Precompiled headers (PCH) — target_precompile_headers(). Amortizes heavy STL/3rd-party header parsing (fmt, nlohmann, Eigen, ITK) across all TUs.
- Unity builds — CMAKE_UNITY_BUILD=ON. Batches N .cpp files into one TU. Cuts per-file overhead + header re-parsing. Hide behind option; conflicts with some patterns.
- Single CUDA arch for dev — each -arch multiplies nvcc kernel compilation. 3 archs = 3x CUDA time.
- Default tests/benchmarks OFF in dev — dev builds only what changes.
- vcpkg/FetchContent binary caching — avoid rebuilding deps.
- Split debug info (-gsplit-dwarf) — smaller objects, faster link+debug.
Source structure
- Move heavy includes from headers to .cpp — every include in a public header is recompiled by every consumer. The #1 structural lever.
- Forward declarations instead of includes where possible.
- Pimpl for API-stability + compile firewall (bigger refactor).
- Include-what-you-use discipline.
- Reduce template instantiation spread — one TU per heavy template user; explicit instantiation.
Part 2 — Audit of this repo
Verdict: mixed — solid structure (targets, project_options, per-config flags), but heavy headers leak into public API and tests compile twice.
Finding 1: Every test file compiled twice — 30+ separate test executables plus combined binary
What: tests/CMakeLists.txt:38-69 lists all test sources for dolphin_all_tests, while dolphin_add_test already builds 30 individual executables (tests/unit/CMakeLists.txt alone: 14). test_cpu_backend.cpp is compiled in backends/cpu/tests/CMakeLists.txt:3 and tests/CMakeLists.txt:68. Each test TU pulls TestUtils.h → Image3D.h → 5 ITK templates.
Why: ~31 extra compilations per build + 31 link jobs. Largest avoidable cost.
Suggestion: Option-gate the two modes, e.g. option(DOLPHIN_COMBINED_TESTS ...). Default: build only dolphin_all_tests (it already runs everything via gtest_discover_tests). Individual targets become opt-in for debugging. Keep dolphin_add_test for files not in the combined list.
Reference: tests/CMakeLists.txt:38-118, tests/unit/CMakeLists.txt, backends/cpu/tests/CMakeLists.txt:3.
Finding 2: ITK filter headers in public Image3D.h
What: dolphin_image/include/dolphin_image/Image3D.h:18-22 includes itkExtractImageFilter.h, itkRegionOfInterestImageFilter.h, itkImageDuplicator.h — none used in the header's declarations (only itkImage.h + iterators are, for ImageType and Iterator members).
Why: 38 files include Image3D.h; each pays for 3 unneeded ITK template stacks. Same pattern: TiffReader.h:28, TiffWriter.h:22-23, LabeledDeconvolutionExecutor.h:30, Postprocessor.h:19 (only itkImage.h needed at most).
Suggestion: Move filter includes into the corresponding .cpp files. Keep itkImage.h + itkImageRegionIterator.h in Image3D.h (needed by declarations). Mechanical, zero behavior change.
Reference: dolphin_image/include/dolphin_image/Image3D.h:18-22.
Finding 3: No PCH despite spdlog(+bundled fmt) + nlohmann + ITK in nearly every TU
What: Config.h (included transitively by almost everything via SetupConfig.h) includes nlohmann/json.hpp (~25k lines) and spdlog/spdlog.h (pulls bundled fmt). Logging.h pulls 7 spdlog headers. No target_precompile_headers anywhere in project targets.
Why: Every TU re-parses fmt + json + ITK front-end. This is the classic PCH payoff case.
Suggestion: On dolphin, dolphin_image, cpu_backend:
target_precompile_headers(dolphin PRIVATE
    <nlohmann/json.hpp>
    <spdlog/spdlog.h>
    <itkImage.h>
    <itkImageRegionIterator.h>
)
Caveat: with ccache, add ccache -o sloppiness=pch_defines,time_macros or PCH misses cache.
Reference: include/dolphin/Config.h:16-21, include/dolphin/Logging.h:15-21, CMakeLists.txt:199.
Finding 4: Local dev build hygiene — Unix Makefiles, no ccache, 3 CUDA archs, benchmarks ON
What: Local cache: CMAKE_GENERATOR=Unix Makefiles, no COMPILER_LAUNCHER, ENABLE_BENCHMARKS=ON default, CMAKE_CUDA_ARCHITECTURES="75;80;90" (repo default; every kernel compiled 3x). CI already uses ccache (ci-main.yml:139-140) — local doesn't.
Why: Serial or under-parallel builds, no cross-build caching, 4 benchmark executables built unconditionally.
Suggestion: Add CMakePresets.json:
{
  "version": 6,
  "configurePresets": [{
    "name": "dev",
    "binaryDir": "build-dev",
    "cacheVariables": {
      "CMAKE_BUILD_TYPE": "Release",
      "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
      "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
      "CMAKE_CUDA_ARCHITECTURES": "native",
      "ENABLE_BENCHMARKS": "OFF"
    },
    "generator": "Ninja"
  }]
}
Flip ENABLE_BENCHMARKS/ENABLE_TESTS defaults to OFF, enable in CI explicitly.
Reference: CMakeLists.txt:106-107,135, build/CMakeCache.txt:43, AGENTS.md build instructions (plain make).
Finding 5: Duplicate add_subdirectory(lib/cube) in CUDA backend
What: backends/cuda/CMakeLists.txt:1 and line 14 both call add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/cube ...).
Why: Line 14 sits after the CUDAToolkit_FOUND guard and return() — so line 1 runs unconditionally before the guard exists, defeating it. Depending on CMake version this either errors or silently double-processes the CUBE directory.
Suggestion: Delete line 1. Keep the guarded one at line 14 so CUBE is only added when toolkit is found.
Reference: backends/cuda/CMakeLists.txt:1,14.
Priority order
1. Test dedup (Finding 1) — biggest absolute win
2. Presets + ccache + Ninja (Finding 4) — free, ~zero risk
3. ITK header hygiene (Finding 2) — mechanical
4. PCH (Finding 3) — do after 2, measure
5. CUBE duplicate (Finding 5) — 1-line fix
