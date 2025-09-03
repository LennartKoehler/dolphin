# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

DOLPHIN (Deconvolution with Optimized Local PSFs for High-speed Image recoNstruction) is a C++ scientific computing application for microscopy image deconvolution with both CLI and GUI frontends.

## Build Commands

### Prerequisites
- CUDA toolkit 12.1+ (for GPU version)
- OpenCV 4.6.0+
- FFTW 3.3.10+
- LibTIFF 4.7.0+
- OpenMP

### Build Process
```bash
# Build CUBE library (required for GPU support)
mkdir ./lib/cube/build
cd ./lib/cube/build
cmake ..
make

# Build main application
mkdir ./build
cd ./build
cmake ..
make
```

### Executables
- `./dolphin` - CPU version
- `./dolphincuda` - GPU version (if CUDA available)

### Testing
No formal test suite exists. Use test data in configs/default_config.json path.

## Critical Non-Obvious Patterns

### Configuration System
- All configuration classes inherit from `Config` base class
- Use `readParameter<T>()` for required fields, `readParameterOptional<T>()` for optional fields
- JSON loading automatically uses config paths relative to executable location
- `SetupConfig::createFromJSONFile()` is the canonical way to load configurations

### Algorithm Factory Pattern
- Deconvolution algorithms registered via `DeconvolutionAlgorithmFactory::getInstance()`
- Register algorithms with factory pattern in constructor, not at runtime
- Algorithm names must exactly match CLI arguments ("RichardsonLucy", "RegularizedInverseFilter", etc.)

### Memory Management
- Uses `std::shared_ptr` extensively for PSFs and algorithms
- PSFManager handles PSF lifecycle - prefer using its factory methods
- Hyperstack data structures use channels for multi-dimensional image data

### GPU Support Architecture
- CUDA code compiled separately via `lib/cube/CMakeLists.txt`
- Conditional compilation with `#ifdef CUDA_AVAILABLE` guards
- GPU version links with `CUBE` library - must build cube first or compilation fails

### File I/O Conventions
- TIFF reading auto-detects file vs directory based on extension
- Results always saved to `../result/` directory relative to executable
- Separate layer saving controlled by `sep` flag (not `seperate`)

### Frontend Architecture  
- Dual frontend system: `CLIFrontend` for CLI, `GUIFrontend` for GUI
- Frontends selected based on `argc > 1` in main()
- GUI passes `Dolphin` instance directly to constructor (unusual dependency injection)

### PSF System
- PSFs can be applied globally OR to specific subimages/layers via arrays
- Position conflicts resolved by PSF array index (lower index wins)
- PSF safety border critical for convolution algorithm correctness
- GibsonLanni and Gaussian PSF generators have different coordinate systems

### Critical Build Flags
- `CMAKE_CXX_FLAGS`: `-O3 -ffast-math -fopenmp -march=native` (CPU optimization)
- `CUDA_ARCHITECTURES`: "75;80;90" (explicit GPU targets)
- Uses OpenMP for parallelization - compile with `-fopenmp`

## Avoid Common Pitfalls

### Configuration Loading
- JSON paths are relative to CMAKE_BINARY_DIR, not project root
- Missing required parameters throw `std::runtime_error` with field name
- Deconvolution section must exist in JSON, even if using defaults

### Algorithm Registration
- Adding new algorithms requires updating both factory register list AND CLI option parsing
- Algorithm names case-sensitive and must match exactly
- Default algorithm is "RichardsonLucyTotalVariation"

### PSF Coordinate System
- GibsonLanni uses different coordinate conventions than Gaussian
- PSF dimensions ordered [x, y, z] but sometimes referenced as [z, y, x] in code
- Centering calculations vary by PSF model type

### Memory Debugging
- Hyperstacks use copy constructor extensively for data copying
- FFT operations use external FFTW library, not custom implementations
- Large datasets benefit from subimage processing but requires careful PSF mapping

## Performance Notes

- OpenMP enabled by default - compile with `-fopenmp` for parallelization
- GPU version significantly faster but requires CUDA 12.1+ and specific GPU architectures
- Subimage processing controlled by `subimageSize` parameter (0 = auto-adjust to PSF size)
- Final image always saved regardless of errors - check console for algorithm failures

## Development Patterns

### Adding New Algorithms
1. Extend `BaseDeconvolutionAlgorithm`
2. Register in factory constructor
3. Add CLI option in frontend
4. Update default configurations

### GUI Development
- ImGui-based interface with ImPlot3D for 3D visualization
- Spinning donut animation in `SpinningDonut.cpp` for loading indicator
- Window hierarchy uses `MainWindow` -> `ConfigWindow` -> `BackendConfigWindow` pattern

### Code Style
- Header implementations in separate .cpp files (unusual for C++ projects)
- Factory pattern heavily used throughout
- Configuration parameters use Hungarian notation in structs (naming violates modern C++ conventions but is project-specific)