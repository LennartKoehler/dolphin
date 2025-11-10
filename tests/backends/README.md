# Backend FFT Performance Test

This test compares the performance of FFT operations between two approaches:
1. One huge FFT operation
2. 1000 smaller FFT operations with the same total data size

## Purpose

The goal is to understand whether splitting large FFT operations into smaller ones provides performance benefits, which can be important for optimizing deconvolution algorithms.

## How It Works

The test:
1. Creates a total data volume (64x64x64 = 262,144 elements)
2. Compares:
   - One FFT of size ~64x64x64 (actual size adjusted to be a perfect cube)
   - 1000 FFTs of smaller size (each ~6x6x6 = 216 elements)
3. Measures and compares execution times
4. Reports performance ratios

## Building

The test is automatically built when you build the project with the CMake configuration. It will be linked against both CPU and CUDA backends (if available).

## Running

### Method 1: Using CTest (Recommended)

```bash
# Build the project
mkdir build && cd build
cmake ..
make

# List all available tests
ctest --show-only

# Run the FFT performance test specifically
ctest -R "FFTPerformanceTest"

# Run all backend tests
ctest -R "BackendTest"
```

### Method 2: Direct Execution

```bash
# Build the project
mkdir build && cd build
cmake ..
make

# Run the FFT performance test directly
./fft_performance_test

# Run other backend tests directly
./cpu_test
./cuda_test  # Only if CUDA is available
```

## Expected Output

The test will output:
- Configuration details (FFT sizes, number of operations)
- Execution time for the huge FFT
- Total and average execution time for the small FFTs
- Performance comparison ratios

## Test Structure

The project now has individual test executables:
- `cpu_test` - Tests CPU backend initialization
- `cuda_test` - Tests CUDA backend initialization (if available)
- `fft_performance_test` - Runs FFT performance comparison
- `dolphin_tests` - Combined test with all functionality

All test executables will be located in the build directory after building.

## Dependencies

- CPU backend: Built with FFTW
- CUDA backend: Built with cuFFT (if CUDA is available)

## Notes

- The test uses random data to ensure realistic performance measurements
- Memory is properly allocated and freed for each test
- Both backends are tested if available
- The test handles cases where backends are not built or CUDA is not available
- Individual test executables allow for more targeted testing and debugging