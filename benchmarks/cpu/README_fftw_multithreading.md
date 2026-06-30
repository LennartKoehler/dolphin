# FFTW Multithreading Performance Test

This test evaluates the performance of FFTW (Fastest Fourier Transform in the West) library when using multiple threads for 3D FFT operations.

## Features

- **Multithreading Performance Analysis**: Tests FFTW performance with different thread counts (1, 2, 4, 8 threads)
- **Scaling Efficiency Metrics**: Calculates speedup and efficiency percentages for each thread configuration  
- **Memory Bandwidth Testing**: Measures effective memory bandwidth during FFT operations
- **Multiple Problem Sizes**: Tests with various 3D FFT dimensions to analyze scaling behavior

## What the Test Does

1. **Initialization**: Sets up FFTW threading support and configures OpenMP
2. **Problem Size Testing**: Runs tests on different 3D FFT sizes:
   - 64×64×64 (Medium)
   - 128×128×128 (Large) 
   - 256×256×128 (Very Large)
3. **Threading Analysis**: For each problem size:
   - Tests performance with 1, 2, 4, and 8 threads (limited by available CPU cores)
   - Measures execution time and calculates speedup vs single-threaded performance
   - Computes threading efficiency as a percentage
4. **Memory Bandwidth**: Estimates memory throughput during FFT operations

## Output

The test provides detailed output including:
- Execution times for each thread count
- Speedup ratios compared to single-threaded performance
- Threading efficiency percentages
- Memory bandwidth estimates
- Performance scaling analysis

## Building and Running

The test is automatically built when FFTW libraries are available. To run:

```bash
cd build
make fftw_multithreading_test
./fftw_multithreading_test
```

## Requirements

- FFTW3 library with threading support (`libfftw3-dev` and `libfftw3-3`)
- OpenMP support
- C++17 compiler

## Understanding Results

- **Speedup**: Ratio of single-threaded time to multi-threaded time (higher is better)
- **Efficiency**: Speedup divided by number of threads × 100% (closer to 100% is better)
- **Ideal Scaling**: Linear speedup would show efficiency of 100% for all thread counts
- **Real-world Performance**: Efficiency typically decreases with more threads due to overhead and memory bandwidth limitations
