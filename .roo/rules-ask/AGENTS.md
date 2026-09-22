# Ask Mode Rules - DOLPHIN Project

## Project Architecture Context

### Core System Design
DOLPHIN is a microscopy image deconvolution application with dual architecture: CLI for batch processing and GUI for interactive use. The system uses factory pattern extensively for algorithms and conditional compilation for GPU support. Key architectural insight: the project separates scientific computing cores from presentation layers through a facade pattern.

### Configuration System
All configuration classes inherit from `Config` base class which uses template-based parameter loading. This is non-obvious because it handles both required and optional fields automatically through `readParameter<T>()` and `readParameterOptional<T>()` methods. Importantly, JSON paths are relative to executable location, not project root.

### Memory Management Pattern
The project uses `std::shared_ptr` extensively but with specific conventions: PSFs and algorithms are managed through factory methods that return shared_ptr, preventing direct object instantiation. Hyperstack processing uses copy constructors extensively rather than move semantics, which can be surprising for performance-focused developers.

### GPU vs CPU Architecture
The CUDA implementation is compiled separately via `lib/cube` subproject with `#ifdef CUDA_AVAILABLE` guards. This creates a non-obvious dependency: GPU version must build the cube library first and uses hard-coded GPU architectures ("75;80;90"). The interface appears unified but implementations are completely separate.

## Critical Non-Obvious Patterns

### Algorithm Registration System
Algorithms are registered in factory constructor using lambda functions, not at runtime. The registration names must exactly match CLI arguments (case-sensitive), and adding new algorithms requires updating BOTH the factory registration AND the CLI option parsing. This creates a hidden coupling that's not obvious from the interface.

### PSF Management Complexity
PSF (Point Spread Function) system supports global application OR position-specific application via arrays. The conflict resolution rule (lower index wins) is non-obvious because it silently ignores position conflicts rather than throwing errors. PSF coordinate systems vary by generator type - GibsonLanni uses different conventions than Gaussian which causes confusion.

### File I/O Detection Logic
TIFF reading auto-detects file vs directory based on extension, which works automatically but the implementation uses string manipulation: `imagePath.substr(imagePath.find_last_of(".") + 1) == "tif"`. Results are always saved to `../result/` relative to executable location, not project root.

### Frontend Selection Logic
The frontend selection in `main()` is based on `argc > 1` - CLI arguments use CLIFrontend, no arguments use GUIFrontend. This is non-obvious because GUIFrontend receives the Dolphin instance directly as a parameter, creating an unusual dependency injection pattern.

## Performance Characteristics

### OpenMP Usage
The project compiles with `-fopenmp` for parallelization, but this isn't obvious from the source code alone. Performance is heavily dependent on correct OpenMP flags and hyperthreading support. The optimization flags `-O3 -ffast-math -march=native` can cause precision differences between debug and release builds.

### Subimage Processing
When `subimageSize` parameter is 0, the system auto-adjusts to PSF size. This adaptive behavior is convenient but means performance characteristics change based on PSF dimensions. Manual subimage sizing requires careful PSF position mapping to avoid artifacts.

### Memory Allocation Patterns
Large datasets benefit from subimage processing but the implementation creates temporary memory copies during FFT operations. The GPU version uses separate memory pools from the CPU version, with no unified memory management visible at the application level.

## Development Context

### Code Structure Patterns
Despite being a C++ project, headers contain minimal implementation with all logic in separate .cpp files. This is unusual for C++ where header implementations are common. Configuration parameters use Hungarian notation (violating modern C++ conventions) but this is consistent throughout the project.

### Testing Framework
No formal test suite exists - testing relies on configuration files in `configs/` directory and manual verification with test data in `configs/default_config.json`. This makes regression testing challenging for algorithm changes.

### Build Dependencies
The build process requires specific order: CUBE library before main application. Missing this dependency causes cryptic compilation errors. CUDA support requires specific toolkit version (12.1+) and compatible GPU architectures.

### GUI System Integration
The GUI uses ImGui with SpinningDonut.cpp for loading indicators and a window hierarchy that goes MainWindow -> ConfigWindow -> BackendConfigWindow. The GUI passes Dolphin instances directly to constructors, creating tight coupling not typical for modern GUI architecture.

## Common Misconceptions

### Algorithm Naming
Developers often assume algorithm names are configurable strings, but they're hard-coded constants in the factory constructor. New algorithms require code changes, not just configuration updates.

### Configuration Loading
JSON loading appears to be standard but uses relative paths and has strict requirements about section existence (Deconvolution section must exist even when using defaults).

### GPU Performance
GPU version provides 5-10x speedup but uses different numerical precision due to optimized kernels, which can cause result differences between CPU andGPU versions.

### File Format Support
Despite supporting multiple TIFF extensions, the detection logic is fragile with Unicode paths and specific extension cases that can cause silent failures.