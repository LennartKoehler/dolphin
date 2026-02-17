#pragma once
#include "CUBE.h"

// Mat operations
__global__ void complexMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexScalarMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t B, complex_t* C);
__global__ void complexElementwiseMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatMulConjugateGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexElementwiseMatDivGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__global__ void complexElementwiseMatDivStabilizedGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__global__ void complexAdditionGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__global__ void complexAdditionGlobal(int Nx, int Ny, int Nz, complex_t** data, complex_t* sums, int N);
__global__ void sumToOneReal(int Nx, int Ny, int Nz, complex_t** data, int nImages, int imageVolume);

// Regularization
__global__ void calculateLaplacianGlobal(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);
__global__ void gradientXGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX);
__global__ void gradientYGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY);
__global__ void gradientZGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ);
__global__ void computeTVGlobal(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv);
__global__ void normalizeTVGlobal(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon);

// Tiled
__global__ void calculateLaplacianTiledGlobal(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);

// Fourier Shift
__global__ void normalizeDataGlobal(int Nx, int Ny, int Nz, complex_t* d_data);
__global__ void octantFourierShiftGlobal(int Nx, int Ny, int Nz, complex_t* data);
__global__ void padMatGlobal(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat, int offsetX, int offsetY, int offsetZ);

// Device Kernels
__global__ void deviceTestKernelGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__device__ void complexMatMulDevice(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__device__ void complexElementwiseMatMulDevice(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C);
__device__ void complexElementwiseMatDivDevice(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon);
__device__ void calculateLaplacianDevice(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);
__device__ void gradientXDevice(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX);
__device__ void gradientYDevice(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY);
__device__ void gradientZDevice(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ);
__device__ void computeTVDevice(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv);
__device__ void normalizeTVDevice(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon);
__global__ void padMatDevice(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat, int offsetX, int offsetY, int offsetZ);
__device__ void calculateLaplacianTiledDevice(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft);