#pragma once
#include "CUBE.h"


namespace CUBE_MAT {

    cudaError_t complexMatMul(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatMul(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatMulConjugate(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatDiv(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatDivStabilized(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream = 0);
    cudaError_t complexScalarMul(int Nx, int Ny, int Nz, complex_t* A, complex_t scalar , complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexAddition(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexAddition(complex_t** data, complex_t* sums, int nImages, int imageVolume, cudaStream_t stream = 0);
    cudaError_t sumToOneReal(complex_t** data, int nImages, int imageVolume, cudaStream_t stream = 0);
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacian(int Nx, int Ny, int Nz, complex_t* psf, complex_t* laplacian_fft, cudaStream_t stream = 0);
    cudaError_t gradX(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX, cudaStream_t stream = 0);
    cudaError_t gradY(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY, cudaStream_t stream = 0);
    cudaError_t gradZ(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ, cudaStream_t stream = 0);
    cudaError_t computeTV(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv, cudaStream_t stream = 0);
    cudaError_t normalizeTV(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon, cudaStream_t stream = 0);
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianTiled(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);
}

namespace CUBE_FTT {
    // Fourier Shift, Padding and Normalization
    cudaError_t octantFourierShift(int Nx, int Ny, int Nz, complex_t* data, cudaStream_t stream = 0);
    cudaError_t padMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat);
    cudaError_t normalizeData(int Nx, int Ny, int Nz, complex_t* d_data, cudaStream_t stream = 0);
}

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    cudaError_t deviceTestKernel(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
}



