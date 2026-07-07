#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#ifdef DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

typedef real_t complex_t[2];
// Mat operations
__global__ void complexMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexScalarMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, real_t realB, real_t imagB, complex_t* C);
__global__ void complexElementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatMulConjugateGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatDivGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__global__ void complexElementwiseMatDivStabilizedGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__global__ void complexAdditionGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexAdditionGlobal(complex_t** data, complex_t* sums, size_t nImages, size_t imageVolume);

__global__ void elementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C);
__global__ void scalarMulGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideC, real_t* A, real_t B, real_t* C);
__global__ void elementwiseMatDivGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, real_t epsilon);

__global__ void sumGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* data, complex_t* result);
__global__ void meanSquareErrorGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* a, complex_t* b, real_t* result);
__global__ void sumToOneGlobal(real_t** data, size_t nImages, size_t imageVolume);

// Regularization
__global__ void calculateLaplacianGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft);
__global__ void gradientXGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradX);
__global__ void gradientYGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradY);
__global__ void gradientZGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradZ);
// Gradient kernels for real-valued data
__global__ void gradientXGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX);
__global__ void gradientYGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradY);
__global__ void gradientZGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradZ);
// Combined gradient kernel (computes all three gradients in a single pass)
__global__ void gradientGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, real_t* gradY, real_t* gradZ);
// Divergence kernels (backward differences — adjoint of forward gradient)
__global__ void divergenceGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideGx, size_t strideGy, size_t strideGz, size_t strideOut, real_t* gx, real_t* gy, real_t* gz, real_t* result);
__global__ void divergenceGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* result);
__global__ void computeTVGlobal(size_t Nx, size_t Ny, size_t Nz, real_t lambda, complex_t* div, complex_t* tv);
__global__ void computeTVGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideDiv, size_t strideTv, real_t lambda, real_t* div, real_t* tv);
__global__ void normalizeTVGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t beta);
__global__ void normalizeTVGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideGradX, size_t strideGradY, size_t strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t beta);

// Tiled
__global__ void calculateLaplacianTiledGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft);

// Fourier Shift
__global__ void normalizeDataGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* d_data);
__global__ void octantFourierShiftGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* data);
__global__ void octantFourierShiftGlobal(size_t Nx, size_t Ny, size_t Nz, size_t stride, real_t* data);
__global__ void padMatGlobal(size_t oldNx, size_t oldNy, size_t oldNz, size_t newNx, size_t newNy, size_t newNz, complex_t* oldMat, complex_t* newMat, size_t offsetX, size_t offsetY, size_t offsetZ);
