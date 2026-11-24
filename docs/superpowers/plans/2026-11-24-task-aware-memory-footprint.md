# Task-Aware Memory Footprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread the plan's task (cube) count through the `MemoryComponent::footprint()` API so per-thread systems report `min(threads, tasks)` instances — a 1-task plan never models multiple input/output buffer sets.

**Architecture:** The virtual `footprint(const CuboidShape&)` gains a `numTasks` parameter. The three strategy-owned per-thread components (io staging, algorithm working set, padding peak) take their thread count in their constructor and internally report `min(threads, numTasks)` instances; `MemoryFootprint::add()` counts drop to 1 in production. The splitter learns each candidate's cube count by splitting before the budget check.

**Tech Stack:** C++17, CMake presets (`dev`), GoogleTest (combined `dolphin_all_tests` binary).

**Spec:** In-conversation design (2026-11-24), approved by user: min lives *inside the components*; padding peak capped for consistency; `UNLIMITED_TASKS` (SIZE_MAX) = no cap = old behavior.

## Global Constraints

- Build/test: `cmake --preset dev && cmake --build --preset dev`, run tests with `ctest --preset dev` (or the combined binary with `--gtest_filter`).
- `footprint()` stays pure and cheap; only uses info the owner already has (MemoryComponent.h contract).
- No default arguments on the virtual `footprint()` (static-dispatch trap). Call sites pass `numTasks` explicitly.
- `MemoryFootprint::add(component, count)` count machinery stays (used by tests, generic API); production adds use count 1.

## Review Focus

- Single-cube images: budget check at full-image size must use `numTasks == 1` (1 staging set, 1 working set) — pinned by the recording-fake test in Task 3.
- `numTasks > threads` must not increase demand beyond thread count — pinned by Task 2 unit tests.
- Fixed `cubeSize` path must check the budget with the real cube count, not a guess — pinned by Task 3 reorder (existing fixed-size behavior tests must stay green).
- `numTasks` unknown (minShape pre-check, generic tests) must reproduce today's numbers exactly — pinned by UNLIMITED_TASKS call sites + existing memory_test expectations.

---

### Task 1: Add `numTasks` to the footprint API (behavior-preserving)

**Files:**
- Modify: `backends/dolphinbackend/include/dolphinbackend/MemoryComponent.h`
- Modify: `backends/dolphinbackend/include/dolphinbackend/MemoryFootprint.h`, `backends/dolphinbackend/src/MemoryFootprint.cpp`
- Modify (signature bump only, param unused): `dolphin_image/include/dolphin_image/IO/TiffReader.h`, `dolphin_image/include/dolphin_image/IO/TiffWriter.h`, `include/dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h`, `backends/cpu/include/cpu_backend/CPUBackendManager.h`, `backends/cuda/include/cuda_backend/CUDABackendManager.h`, `include/dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h`
- Modify (call sites): `src/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.cpp`, `src/deconvolution/deconvolutionStrategies/ImageSplit.cpp`
- Modify (fakes/call sites): `tests/deconvolution/deconvolutionStrategies/memory_test.cpp`, `tests/deconvolution/deconvolutionStrategies/imageSplit_test.cpp`, `tests/deconvolution/deconvolutionStrategies/imageSplit_contract_test.cpp`, `tests/integration/peak_memory_test.cpp`, `tests/IO/read_write_test.cpp`

**Interfaces:**
- Produces: `MemoryComponent::footprint(const CuboidShape& cubeShape, size_t numTasks) const` (pure virtual); `MemoryComponent::UNLIMITED_TASKS = std::numeric_limits<size_t>::max()`; `MemoryFootprint::total(const CuboidShape&, size_t numTasks)`, `MemoryFootprint::breakdown(const CuboidShape&, size_t numTasks)`.

- [ ] **Step 1: Change the signatures** — add `size_t numTasks` after `cubeShape` in the virtual, `total`, `breakdown`; forward `numTasks` from `total`/`breakdown` to each component; add `UNLIMITED_TASKS` constant + contract doc sentence: numTasks = total cubes in the plan; per-thread systems never report more instances than tasks.
- [ ] **Step 2: Bump every override and call site** — unnamed `size_t` param where ignored; all existing call sites pass `MemoryComponent::UNLIMITED_TASKS` (createPlan lines 148-150, ImageSplit line 262, getCubes lines 278/288, all test call sites). Counts in `buildMemoryFootprint` unchanged.
- [ ] **Step 3: Build + run footprint tests** — `cmake --build --preset dev && ctest --preset dev -R "memory|imageSplit|peak_memory|read_write"`. Expected: all PASS (numbers unchanged).
- [ ] **Step 4: Commit** — `git commit` as `refactor: add numTasks parameter to memory footprint API`.

### Task 2: `min(threads, numTasks)` inside the strategy components

**Files:**
- Modify: `include/dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h` (IOStagingMemoryComponent, AlgorithmWorkingSetMemoryComponent, PaddingPeakMemoryComponent)
- Modify: `src/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.cpp` (buildMemoryFootprint: thread counts into constructors, count-1 adds)
- Test: `tests/deconvolution/deconvolutionStrategies/memory_test.cpp`
- Modify: `tests/integration/peak_memory_test.cpp` (totals still hold with numTasks >= threads; drop `(x2)`/`(x3)` breakdown assertions)

**Interfaces:**
- Consumes: Task 1 signatures.
- Produces: `IOStagingMemoryComponent(size_t ioThreads)`, `AlgorithmWorkingSetMemoryComponent(const DeconvolutionAlgorithm&, bool onDevice, size_t workerThreads)`, `PaddingPeakMemoryComponent(const Padding&, size_t ioThreads)` — each reporting `min(threads, numTasks)` instances.

- [ ] **Step 1: Write failing tests** in memory_test.cpp: staging (ioThreads=4): `footprint(shape, 1)` = 1x {vol, 2vol}, `(shape, 2)` = 2x, `(shape, 8)` = 4x; algorithm (workerThreads=3, onDevice=false): `(shape, 1)` = 1x workingSet, `(shape, 10)` = 3x.
- [ ] **Step 2: Run** `--gtest_filter=MemoryTest.*Staging*:*WorkingSet*` → FAIL (compile: no matching ctor).
- [ ] **Step 3: Implement** the three components + buildMemoryFootprint rewiring.
- [ ] **Step 4: Update peak_memory_test** (pass `UNLIMITED_TASKS` to total/breakdown; remove the two `(x`-marker assertions), run full suite. Expected: PASS.
- [ ] **Step 5: Commit** — `feat: cap per-thread footprint instances by task count`.

### Task 3: Splitter and call sites pass the real task count

**Files:**
- Modify: `src/deconvolution/deconvolutionStrategies/ImageSplit.cpp` (splitImageHomogeneous: split first, then `footprint.total(currentMaxSize, cubePositions.size())`)
- Modify: `src/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.cpp` (getCubes fixed-size branch: split then check; minShape pre-check keeps UNLIMITED_TASKS; createPlan line 148-150: real `cubeCoordinatesWithPadding.size()`)
- Test: `tests/deconvolution/deconvolutionStrategies/imageSplit_test.cpp`

**Interfaces:**
- Consumes: Tasks 1-2.

- [ ] **Step 1: Write failing test**: recording fake (`lastNumTasks` captured in footprint) — single-cube image (image fits whole) must observe `numTasks == 1` on the first `total()` call.
- [ ] **Step 2: Run** → FAIL (observed UNLIMITED_TASKS).
- [ ] **Step 3: Restructure the search loop**: `addCubeRecursion` every iteration (exception → fail return, same as today); break when `size >= minNumberCubes && budget.fits(footprint.total(currentMaxSize, cubePositions.size()))`; fail message keeps last candidate's breakdown.
- [ ] **Step 4: getCubes fixed-size branch**: call `splitImageHomogeneousFixedSize` first, then budget-check `footprint.total(fixedPadded, cubes.size())` (same error message); createPlan passes the real count.
- [ ] **Step 5: Run full suite** `ctest --preset dev`. Expected: all PASS.
- [ ] **Step 6: Commit** — `feat: size cubes with the candidate's true task count`.
