# cuFFT vs FFTW Performance Comparison Test

This directory contains a standalone test that compares the performance of cuFFT (CUDA FFT) with FFTW (Fastest Fourier Transform in the West) for 3D FFT operations.

## Files

- `cufft_vs_fftw_comparison.cpp` - Main test implementation
- `CMakeLists.txt` - Build configuration (updated to include this test)
- `README_cufft_vs_fftw.md` - This documentation file

## Requirements

### System Requirements
- CUDA 11.0 or later
- CUDA Toolkit with cuFFT library
- FFTW library (single precision, `fftw3f`)

### Library Dependencies
- CUDA Runtime API
- cuFFT library
- FFTW library (fftw3)

## Building

### Prerequisites

1. **Install FFTW** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libfftw3-dev
   
   # Or build from source
   wget http://www.fftw.org/fftw-3.3.10.tar.gz
   tar -xvzf fftw-3.3.10.tar.gz
   cd fftw-3.3.10
   ./configure --enable-single --enable-threads
   make
   sudo make install
   ```

2. **Ensure CUDA is installed** and available in your system path.

### Build Steps

1. Install required dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cuda-toolkit libfftw3-dev pkg-config
   
   # CentOS/RHEL
   sudo yum install cuda-toolkit fftw-devel pkgconfig
   ```

2. Navigate to the build directory:
   ```bash
   cd build
   ```

3. Configure CMake with CUDA support:
   ```bash
   cmake .. -DCMAKE_CUDA_ARCHITECTURES=75;80;90
   ```

4. Build the test:
   ```bash
   make cufft_vs_fftw_comparison
   ```

## Running the Test

Execute the test from the build directory:
```bash
./tests/cufft_vs_fftw_comparison
```

### Expected Output

The test will run multiple FFT configurations and output performance comparisons:

```
cuFFT vs FFTW Performance Comparison
=====================================

--- Testing 32x32x32 ---
FFTW 3D FFT (32x32x32) time: 12345 microseconds
cuFFT 3D FFT (32x32x32) time: 6789 microseconds
Validating results... PASSED

--- Testing 64x64x64 ---
FFTW 3D FFT (64x64x64) time: 98765 microseconds
cuFFT 3D FFT (64x64x64) time: 54321 microseconds
Validating results... PASSED

... (additional test sizes) ...

Comparison completed!
```

## Test Configuration

The test automatically runs with the following 3D FFT sizes:
- 32×32×32 (small)
- 64×64×64 (medium) 
- 128×128×128 (large)
- 256×256×256 (very large)

Each test runs 100 iterations to get reliable timing measurements.

## Validation

The test includes a validation step that compares the results from cuFFT and FFTW to ensure they produce equivalent outputs within a numerical tolerance (1e-3).

## Performance Analysis

The test measures:
- Execution time for each FFT library
- Memory allocation and transfer overhead (for cuFFT)
- Plan creation time (separately measured in practice)

## Optimizations Applied

### cuFFT Optimizations
- **Page-locked host memory**: Uses `cudaHostAlloc` for faster host-device memory transfers
- **Asynchronous memory copies**: Overlaps data transfers with computation using `cudaMemcpyAsync`
- **Advanced planning**: Attempts to use `cufftPlanMany` for better performance, falls back to `cufftPlan3d` if needed
- **Stream usage**: Sets optimal CUDA stream configuration
- **Device selection**: Uses device 0 for optimal performance
- **Memory management**: Proper cleanup of all allocated resources

### FFTW Optimizations
- **Multi-threading**: Uses `fftwf_init_threads()` and all available CPU cores
- **Wisdom planning**: Uses `FFTW_MEASURE` flag for optimal performance (takes longer to plan but faster execution)
- **Warm-up run**: Includes a warm-up execution to avoid initialization overhead in timing
- **Thread affinity**: Sets maximum number of threads for parallel processing

### General Optimizations
- **Warm-up phase**: Both libraries perform warm-up runs to ensure fair comparison
- **OpenMP utilization**: Uses all available CPU cores for FFTW
- **Memory efficiency**: Optimized memory allocation patterns for both libraries

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA is installed and in your PATH
2. **FFTW not found**: Install FFTW development libraries
3. **Architecture mismatch**: Specify your GPU architecture with `-DCMAKE_CUDA_ARCHITECTURES`
4. **Library linking errors**: Check that both single-precision FFTW and cuFFT are available

### Build Commands for Specific Systems

**Ubuntu/Debian:**
```bash
sudo apt-get install cuda-toolkit-12-3 libfftw3-dev
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75;80;90
make cufft_vs_fftw_comparison
```

**CentOS/RHEL:**
```bash
sudo yum install cuda-toolkit fftw-devel
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75;80;90
make cufft_vs_fftw_comparison
```

## Notes

- This test is completely standalone and does not depend on any other parts of the DOLPHIN project
- Results will vary depending on your specific GPU hardware
- For best performance, ensure your GPU drivers are up to date
- The test uses single-precision floating point (float) for both libraries