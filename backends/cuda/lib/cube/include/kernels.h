#pragma once
#include <cuComplex.h>
#include <cufft.h>
#include <cufftw.h>

// Conversions
__global__ void fftwToCuComplexKernelGlobal(int Nx, int Ny, int Nz,cuComplex* cuArr, fftw_complex* fftwArr);
__global__ void cuToFftwComplexKernelGlobal(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cuComplex* cuArr);
__global__ void fftwToCufftComplexKernelGlobal(int Nx, int Ny, int Nz, cufftComplex* cufftArr, fftw_complex* fftwArr);
__global__ void cufftToFftwComplexKernelGlobal(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cufftArr);

// Mat operations
__global__ void complexMatMulFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
__global__ void complexMatMulCuComplexGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__global__ void complexElementwiseMatMulCuComplexGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__global__ void complexElementwiseMatDivCuComplexGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__global__ void complexElementwiseMatMulCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__global__ void complexElementwiseMatMulConjugateCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__global__ void complexElementwiseMatDivCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
__global__ void complexElementwiseMatDivNaiveCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__global__ void complexElementwiseMatDivStabilizedCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);

// New FFTW kernels
__global__
void octantFourierShiftFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* data);
__global__
void complexElementwiseMatDivStabilizedFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);
__global__
void complexElementwiseMatMulConjugateFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
__global__
void complexElementwiseMatMulFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
__global__
void complexElementwiseMatDivFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);

// Regularization
__global__ void calculateLaplacianCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);
__global__ void gradientXCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
__global__ void gradientYCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
__global__ void gradientZCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
__global__ void computeTVCufftComplexGlobal(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
__global__ void normalizeTVCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);

// Regularization FFTW
__global__ void calculateLaplacianFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* Afft, fftw_complex* laplacianfft);
__global__ void gradientXFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradX);
__global__ void gradientYFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradX);
__global__ void gradientZFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradX);
__global__ void computeTVFftwComplexGlobal(int Nx, int Ny, int Nz, double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv);
__global__ void normalizeTVFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, double epsilon);


// Tiled
__global__ void calculateLaplacianCufftComplexTiledGlobal(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);

// Fourier Shift
__global__ void normalizeComplexData(int Nx, int Ny, int Nz, cufftComplex* d_data);
__global__ void normalizeFftwComplexDataGlobal(int Nx, int Ny, int Nz, fftw_complex* d_data);
__global__ void octantFourierShiftCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* data);
__global__ void padCufftMatGlobal(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, cufftComplex* oldMat, cufftComplex* newMat, int offsetX, int offsetY, int offsetZ);

// Device Kernels
__global__ void deviceTestKernelGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexMatMulCuComplexDevice(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexElementwiseMatMulCuComplexDevice(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexElementwiseMatDivCuComplexDevice(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexElementwiseMatMulCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__device__ void complexElementwiseMatMulConjugateCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__device__ void complexElementwiseMatDivCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
__device__ void complexElementwiseMatDivNaiveCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__device__ void complexElementwiseMatDivStabilizedCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
__device__ void calculateLaplacianCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);
__device__ void gradientXCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
__device__ void gradientYCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY);
__device__ void gradientZCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ);
__device__ void computeTVCufftComplexDevice(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
__device__ void normalizeTVCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);
__global__ void padCufftMatDevice(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, cufftComplex* oldMat, cufftComplex* newMat, int offsetX, int offsetY, int offsetZ);
__device__ void calculateLaplacianCufftComplexTiledDevice(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);