# FFTW Size Decomposition Test

This test compares the performance of running a single large FFTW with multiple threads versus running multiple smaller FFTWs with the same total size using multiple threads.

## Test Overview

The test evaluates two approaches:

1. **Single Large FFT**: One large 3D FFT using all available threads
2. **Multiple Small FFTs**: Several smaller 3D FFTs running in parallel, distributing threads among them

## Test Scenarios

The test includes several predefined scenarios with the same total data size but different decompositions:

- **128³ vs 8×64³**: One 128×128×128 FFT vs eight 64×64×64 FFTs
- **256×128×64 vs 4×128³**: One large rectangular FFT vs four cubic FFTs  
- **512×64×64 vs 16×64³**: One very elongated FFT vs sixteen cubic FFTs
- **64³ vs 8×32³**: Smaller test case for quick testing

## Key Metrics Measured

- **Execution Time**: Wall clock time for each approach
- **Throughput**: FFTs/operations per second
- **Speedup**: Performance ratio between the two approaches
- **Memory Usage**: Total memory footprint comparison
- **Theoretical Complexity**: O(N log N) analysis for each approach

## Building and Running

### Prerequisites

- FFTW3 library with threading support
- OpenMP
- C++17 compiler
- CMake 3.10+
- Optional: LIKWID for hardware performance monitoring

### Build

```bash
cd build
cmake ..
make fftw_size_decomposition_test
```

### Run

```bash
./fftw_size_decomposition_test
```

### Run with LIKWID (optional)

```bash
# For memory bandwidth analysis
likwid-perfctr -C 0-7 -g MEM_DP -m ./fftw_size_decomposition_test

# For cache analysis  
likwid-perfctr -C 0-7 -g CACHE -m ./fftw_size_decomposition_test

# For floating point operations
likwid-perfctr -C 0-7 -g FLOPS_DP -m ./fftw_size_decomposition_test
```

## Understanding Results

### When Multiple Small FFTs Might Be Faster

- **Better Cache Locality**: Smaller FFTs fit better in CPU cache
- **Reduced Memory Bandwidth Pressure**: Less simultaneous memory access
- **Better Load Distribution**: More parallel work units
- **NUMA Effects**: Better distribution across memory domains

### When Single Large FFT Might Be Faster

- **Lower Overhead**: Less function call and setup overhead
- **Better FFTW Optimization**: FFTW can optimize larger transforms better
- **Less Thread Contention**: Single threaded FFTW execution may be more efficient
- **Memory Access Patterns**: Better spatial locality for very large datasets

### Key Output Sections

1. **Comparison Results**: Shows which approach is faster and by how much
2. **Memory Usage**: Total memory footprint for each approach
3. **Theoretical Analysis**: Computational complexity comparison

## Customization

You can modify the test configurations in `main()` by adding new `TestConfig` entries:

```cpp
TestConfig custom_config = {
    large_dim_x, large_dim_y, large_dim_z,    // Single FFT dimensions
    small_dim_x, small_dim_y, small_dim_z,    // Individual small FFT dimensions
    num_small_ffts,                           // Number of small FFTs
    num_threads,                              // Total threads to use
    num_iterations                            // Timing iterations
};
```

Ensure that `large_dim_x * large_dim_y * large_dim_z == small_dim_x * small_dim_y * small_dim_z * num_small_ffts` for a fair comparison.

## Performance Tips

- Run on a dedicated system without other heavy processes
- Use consistent CPU frequencies (disable frequency scaling)
- Consider NUMA topology for multi-socket systems
- Multiple runs may be needed for stable results
- Monitor CPU temperature during long tests
