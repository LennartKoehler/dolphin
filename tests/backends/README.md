# Backend Tests

Tests for CPU and CUDA backends, including FFT operations and memory management.

## Building

The tests are automatically built when you build the project with `ENABLE_TESTS=ON`:

```bash
mkdir build && cd build
cmake .. -DENABLE_TESTS=ON
make
```

## Running

### Using CTest (Recommended)

```bash
# Run all tests
ctest --output-on-failure

# Run FFT backend tests
ctest -R "FFTBackendTest"

# Run CPU backend tests
ctest -R "cpu_backend_test"
```

### Direct Execution

```bash
# Run individual test executables
./tests/backends/cpu_backend_test
./tests/backends/cuda_backend_test  # Only if CUDA is available
./tests/backends/fft_backend_test
```

## Test Executables

- `cpu_backend_test` - Tests CPU backend operations
- `cuda_backend_test` - Tests CUDA backend operations (if CUDA available)
- `fft_backend_test` - Tests FFT operations and correctness
- `dolphin_all_tests` - Combined test executable with all tests

All test executables are located in the build directory after building.

## Dependencies

- CPU backend: Built with FFTW
- CUDA backend: Built with cuFFT (if CUDA is available)
