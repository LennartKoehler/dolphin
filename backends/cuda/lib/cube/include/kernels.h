#pragma once

#include <cuda_runtime_api.h>

#ifdef DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

typedef real_t complex_t[2];
// Mat operations
__global__ void complexMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexScalarMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t B, complex_t* C);
__global__ void complexElementwiseMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatMulConjugateGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatDivGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__global__ void complexElementwiseMatDivStabilizedGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__global__ void complexAdditionGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexAdditionGlobal(complex_t** data, complex_t* sums, int N, int imageVolume);

__global__ void elementwiseMatMulGlobal(int Nx, int Ny, int Nz, int strideA, int strideB, int strideC, real_t* A, real_t* B, real_t* C);
__global__ void scalarMulGlobal(int Nx, int Ny, int Nz, int strideA, int strideC, real_t* A, real_t B, real_t* C);
__global__ void elementwiseMatDivGlobal(int Nx, int Ny, int Nz, int strideA, int strideB, int strideC, real_t* A, real_t* B, real_t* C, real_t epsilon);

__global__ void sumToOneGlobal(real_t** data, int nImages, int imageVolume);

// Regularization
__global__ void calculateLaplacianGlobal(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);
__global__ void gradientXGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX);
__global__ void gradientYGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY);
__global__ void gradientZGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ);
// Gradient kernels for real-valued data
__global__ void gradientXGlobalReal(int Nx, int Ny, int Nz, int strideIn, int strideOut, real_t* image, real_t* gradX);
__global__ void gradientYGlobalReal(int Nx, int Ny, int Nz, int strideIn, int strideOut, real_t* image, real_t* gradY);
__global__ void gradientZGlobalReal(int Nx, int Ny, int Nz, int strideIn, int strideOut, real_t* image, real_t* gradZ);
__global__ void computeTVGlobal(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv);
__global__ void computeTVGlobalReal(int Nx, int Ny, int Nz, int strideGx, int strideGy, int strideGz, int strideTv, real_t lambda, real_t* gx, real_t* gy, real_t* gz, real_t* tv);
__global__ void normalizeTVGlobal(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon);
__global__ void normalizeTVGlobalReal(int Nx, int Ny, int Nz, int strideGradX, int strideGradY, int strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t epsilon);

// Tiled
__global__ void calculateLaplacianTiledGlobal(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);

// Fourier Shift
__global__ void normalizeDataGlobal(int Nx, int Ny, int Nz, complex_t* d_data);
__global__ void octantFourierShiftGlobal(int Nx, int Ny, int Nz, complex_t* data);
__global__ void octantFourierShiftGlobal(int Nx, int Ny, int Nz, int stride, real_t* data);
__global__ void padMatGlobal(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat, int offsetX, int offsetY, int offsetZ);

