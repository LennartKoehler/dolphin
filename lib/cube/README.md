<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="60" height="60" style="margin-right: 10px;">
    <h1>CUBE - CUDA Boost Engine v0.2.1</h1>
</div>

## Overview

CUBE (CUDA Boost Engine) is a high-performance framework designed for accelerating NxNxN matrix operations and Fourier Transform computations using CUDA and CPP(OpenMP). This framework supports CPU-based, multi-threaded, and GPU-accelerated operations, enabling efficient handling of large complex matrices with advanced matrix multiplication, element-wise operations, and Fourier transformations.

## Key Features

- **Multi-Platform Support**:
   - CPU (Naive C++ and OpenMP)
   - GPU (CUDA)

- **Matrix Operations**:
   - Complex matrix multiplication and element-wise operations (using `fftw_complex`, `cuComplex`, and `cufftComplex` types)
   - Efficient conversion between different complex types (`fftw_complex`, `cuComplex`, `cufftComplex`)

- **Fourier Transforms**:
   - Forward and inverse FFT (cuFFT)
   - Fourier shift for high-frequency centering on both CPU and GPU

- **CUDA Kernels**:
   - Optimized CUDA kernels for matrix multiplication and element-wise operations on complex matrices
   - Support for device-side operations with `__device__` kernels

## Directory Structure

```
CUBE/
├── CMakeLists.txt        # Build configuration
├── include/              # Header files
│   ├── operations.h      # Matrix operations and FFT functions
│   ├── conversions.h     # Conversion functions for complex types
│   └── utils.h           # Utility functions for printing and checking
├── src/                  # Source code
│   ├── main.cpp          # Main program
│   ├── utl.cu            # Utility functions
│   ├── operations.cu     # Matrix operations and FFT implementations
│   └── kernels/          # CUDA kernel implementations
│       ├── global.cu     # Global kernel functions
│       └── device.cu     # Device-specific kernel functions
└── README.md             # Documentation
```

## Build Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository/CUBE.git
   cd CUBE
   ```

2. **Create and navigate to the build directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Generate build files using CMake**:
   ```bash
   cmake ..
   ```

4. **Build the project**:
   ```bash
   make
   ```

5. **Run the executable**:
   ```bash
   ./CUBE
   ```

## Usage

### Matrix Operations

- **CPU Matrix Multiplication (Naive)**:
  ```cpp
  complexMatMultCpp(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C);
  ```

- **CPU Matrix Multiplication (OpenMP)**:
  ```cpp
  complexMatMulCppOmp(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C);
  ```

- **GPU Matrix Multiplication (cuComplex)**:
  ```cpp
  complexMatMulCudaCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C);
  ```

- **GPU Element-wise Matrix Multiplication (cufftComplex)**:
  ```cpp
  complexElementwiseMatMulCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
  ```

### Fourier Transforms

- **Forward FFT (GPU)**:
  ```cpp
  cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan);
  ```

- **Inverse FFT (GPU)**:
  ```cpp
  cufftInverse(cufftComplex* input, cufftComplex* output, cufftHandle plan, int N);
  ```

- **Octant Fourier Shift (GPU)**:
  ```cpp
  octantFourierShiftCufftComplex(int N, cufftComplex* data);
  ```

- **Octant Fourier Shift (CPU)**:
  ```cpp
  octantFourierShiftFftwComplex(int N, fftw_complex* data);
  ```

### Utility Functions

- **Print Matrix**:
  ```cpp
  printFirstElem(fftw_complex* mat);
  printSpecificElem(fftw_complex* mat, int index);
  ```

- **Matrix Initialization**:
  ```cpp
  createFftwUniformMat(int N, fftw_complex* mat);
  createFftwRandomMat(int N, fftw_complex* mat);
  ```

- **Check Matrix Uniformity**:
  ```cpp
  checkUniformity(fftw_complex* mat, int N);
  ```

### Example: Using FFT with CUBE

Here's an example of how to perform element-wise matrix division and Fourier transforms (FFT) using CUBE on a 3D matrix (`N x N x N`) with cuFFT and FFTW. This example demonstrates how to allocate memory on the GPU, perform operations, and calculate the FFT:

```cpp
cufftComplex *d_cua, *d_cub, *d_cuc;
cudaMalloc((void**)&d_cua, N * N * N * sizeof(cufftComplex));
cudaMalloc((void**)&d_cub, N * N * N * sizeof(cufftComplex));
cudaMalloc((void**)&d_cuc, N * N * N * sizeof(cufftComplex));

// Convert FFTW complex data to cuFFT complex format on the device
convertFftwToCufftComplexOnDevice(h_a, d_cua, N);
convertFftwToCufftComplexOnDevice(h_b, d_cub, N);

// Perform element-wise matrix division (cufftComplex)
complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc);

// Optionally, run a test kernel on the device
// deviceTestKernel(N, d_cua, d_cub, d_cuc);

// Create FFT plan for reuse
cufftHandle plan;
cufftResult result = cufftPlan3d(&plan, N, N, N, CUFFT_C2C);
if (result != CUFFT_SUCCESS) {
    std::cerr << "[ERROR] error while creating FFT-Plan: " << result << std::endl;
}

// Perform forward FFT
cufftForward(d_cuc, d_cuc, plan);

// Perform inverse FFT
cufftInverse(d_cuc, d_cuc, plan, N);

// Convert cuFFT complex data back to FFTW format on the host
convertCufftToFftwComplexOnHost(h_c, d_cuc, N);

// Check uniformity and print results
checkUniformity(h_c, N);
printRandomElem(h_c, N);

// Clean up resources
cufftDestroy(plan);
cudaFree(d_cuc);
free(h_c);
```

### Steps in this Example:

1. **Memory Allocation**:
   Allocate memory on the GPU for the complex matrices `d_cua`, `d_cub`, and `d_cuc` using `cudaMalloc`.

2. **Conversion**:
   Convert the input matrices from FFTW format (`h_a`, `h_b`) to cuFFT format on the device using `convertFftwToCufftComplexOnDevice`.

3. **Element-wise Operation**:
   Perform an element-wise matrix division between the two input matrices (`d_cua` and `d_cub`) using `complexElementwiseMatDivCufftComplex`.

4. **FFT Operations**:
   - Create a cuFFT plan (`cufftPlan3d`) for a 3D FFT transformation.
   - Perform a forward FFT using `cufftForward`.
   - Perform an inverse FFT using `cufftInverse`.

5. **Data Conversion**:
   Convert the result from cuFFT back to FFTW format using `convertCufftToFftwComplexOnHost`.

6. **Validation**:
   Check matrix uniformity and print random elements for verification.

7. **Clean-Up**:
   Destroy the FFT plan with `cufftDestroy` and free GPU memory with `cudaFree`.

This process highlights how to integrate cuFFT with CUBE for GPU-accelerated FFT operations while using C++ and CUDA effectively.

### Normalization after cuFFT

In cuFFT, unlike FFTW, automatic normalization by dividing by the number of elements (N) is not performed. To address this, the `normalizeComplexData` function in this framework manually normalizes the inverse cuFFT results in the `cufftInverse` method. After the inverse FFT calculation, this function divides the real part of each complex value by N and sets the imaginary part to 0, ensuring proper scaling and consistency of the results.

Here’s the normalization function used in the framework:

```cpp
__global__
void normalizeComplexData(cufftComplex* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Divide the real part by the number of elements
        d_data[idx].x /= N;

        // Set the imaginary part to 0
        d_data[idx].y = 0;
    }
}
```

## Supported Platforms

- **CUDA**: Requires an NVIDIA GPU with CUDA support.
- **CPU**: Supports both OpenMP-based multi-threading and single-threaded operations.

## Dependencies

- **CUDA Toolkit**: For GPU acceleration and CUDA programming.
- **FFTW**: For CPU-based fftw_complex data handling and operations.
- **OpenMP**: For multi-threading support on the CPU.
- **OpenCV**: For visualization (optional).

## Contributing
- Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
- Icon attribution <a href="https://www.flaticon.com/free-icons/barracuda" title="barracuda icons">Barracuda icons created by Freepik - Flaticon</a>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For inquiries or support, please contact [christoph.manitz@uni-jena.de].

---
